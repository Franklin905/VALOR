from __future__ import print_function
import os
import csv
import math
import time
import wandb
import argparse
import numpy as np
import pandas as pd
import prettytable as pt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader_avvp import LLP_dataset
from main_network import MMIL_Net
from utils.eval_metrics import segment_level, event_level

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def calculate_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm

def lr_warm_up_cos_anneal(optimizer, cur_epoch, warm_up_epoch, max_epoch, lr_min, lr_max):
    if cur_epoch < warm_up_epoch:
        lr = cur_epoch / warm_up_epoch * lr_max
    else:
        lr = (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_epoch-warm_up_epoch)/(max_epoch-warm_up_epoch)*math.pi)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_evaluation_result_table(mode, F_scores_dict):
    f_scores_tb = pt.PrettyTable()
    fields = ["Dataset", "Seg-a", "Seg-v", "Seg-av", "Seg-type", "Seg-event","Event-a", "Event-v", "Event-av", "Event-type", "Event-event"]
    f_scores_tb.field_names = fields
    F_scores_list = [mode] + ['{:.2f}'.format(F_scores_dict[key]) for key in fields if key != 'Dataset']
    f_scores_tb.add_row(F_scores_list)

    return f_scores_tb


def train(args, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = {'total': 0, 'loss_video': 0, 'loss_valor_v': 0, 'loss_valor_a': 0, 'loss_all': 0}

    for batch_idx, batch_data in enumerate(train_loader):
        video_res_feats, video_3d_feats, audios  = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device)
        labels = batch_data['label'].float().to(device)
        audio_pseudo_labels = batch_data['audio_pseudo_labels'].float().to(device)      # (B, 10, 25)
        visual_pseudo_labels = batch_data['visual_pseudo_labels'].float().to(device)    # (B, 10, 25)
        batch_size = video_res_feats.size(0)

        optimizer.zero_grad()
        
        output, _, _, _, frame_logits = model(audios, video_res_feats, video_3d_feats)
        output.clamp_(min=1e-7, max=1 - 1e-7)

        loss_video = criterion(output, labels)
        loss_valor_a = F.binary_cross_entropy_with_logits(frame_logits[:, :, 0], audio_pseudo_labels)
        loss_valor_v = F.binary_cross_entropy_with_logits(frame_logits[:, :, 1], visual_pseudo_labels)

        loss = loss_valor_a + loss_valor_v + loss_video

        train_loss['loss_video'] += (loss_video.item()*batch_size)
        train_loss['loss_valor_v'] += (loss_valor_v.item()*batch_size)
        train_loss['loss_valor_a'] += (loss_valor_a.item()*batch_size)
        train_loss['loss_all'] += (loss.item()*batch_size)
        train_loss['total'] += batch_size

        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        total_grad_norm = calculate_grad_norm(model)
        optimizer.step()

        if args.use_wandb:
            wandb.log({
                'train loss_video': loss_video.item(),
                'train loss_valor_v': loss_valor_v.item(),
                'train loss_valor_a': loss_valor_a.item(),
                'train loss_all': loss.item(),
                'grad_norm': total_grad_norm,
                'iters': (epoch - 1) * len(train_loader) + batch_idx,
                'epoch': epoch
            })

    num_train_data = train_loss['total']
    train_loss = {key: (float(value) / num_train_data) for key, value in train_loss.items() if 'loss' in key}

    return train_loss


def eval(args, model, data_loader, gt_csv_dir, criterion, device):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df_a = pd.read_csv(os.path.join(gt_csv_dir, "AVVP_eval_audio.csv"), header=0, sep='\t')
    df_v = pd.read_csv(os.path.join(gt_csv_dir, "AVVP_eval_visual.csv"), header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    val_loss = {'total': 0, 'loss_video': 0, 'loss_valor_v': 0, 'loss_valor_a': 0, 'loss_all': 0}

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            video_name = batch_data['video_name']
            video_res_feats, video_3d_feats, audios = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device)
            labels = batch_data['label'].float().to(device)
            audio_pseudo_labels = batch_data['audio_pseudo_labels'].float().to(device)   # (B, 10, 25)
            visual_pseudo_labels = batch_data['visual_pseudo_labels'].float().to(device)   # (B, 10, 25)
            batch_size = video_res_feats.size(0)

            output, _, _, frame_prob, frame_logits = model(audios, video_res_feats, video_3d_feats)  # (B, 25), (B, 25), (B, 25), (B, 10, 2, 25)

            # ================= Calculate Validation Loss ===================
            if criterion != None:
                output.clamp_(min=1e-7, max=1 - 1e-7)

                loss_video = criterion(output, labels)
                loss_valor_a = F.binary_cross_entropy_with_logits(frame_logits[:, :, 0], audio_pseudo_labels)
                loss_valor_v = F.binary_cross_entropy_with_logits(frame_logits[:, :, 1], visual_pseudo_labels)

                loss = loss_valor_v + loss_valor_a + loss_video

                val_loss['loss_video'] += (loss_video.item()*batch_size)
                val_loss['loss_valor_v'] += (loss_valor_v.item()*batch_size)
                val_loss['loss_valor_a'] += (loss_valor_a.item()*batch_size)
                val_loss['loss_all'] += (loss.item()*batch_size)
            val_loss['total'] += batch_size
            # ================================================================

            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)  # (B, 25)

            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()  # (10, 25)
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()  # (10, 25)

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)     # (10, 25)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)     # (10, 25)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == video_name[0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels  
            df_vid_v = df_v.loc[df_v['filename'] == video_name[0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    avg_type = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_seg_v))) / 3.
    avg_event = 100 * np.mean(np.array(F_seg))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))

    F_scores = {'Seg-a': 100 * np.mean(np.array(F_seg_a)),
                'Seg-v': 100 * np.mean(np.array(F_seg_v)),
                'Seg-av': 100 * np.mean(np.array(F_seg_av)),
                'Seg-type': avg_type,
                'Seg-event': avg_event,
                'Event-a': 100 * np.mean(np.array(F_event_a)),
                'Event-v': 100 * np.mean(np.array(F_event_v)),
                'Event-av': 100 * np.mean(np.array(F_event_av)),
                'Event-type': avg_type_event,
                'Event-event': avg_event_level}

    num_train_data = val_loss['total']
    val_loss = {key: (float(value) / num_train_data) for key, value in val_loss.items() if 'loss' in key}

    return F_scores, val_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Official Implementation of VALOR')
    parser.add_argument("--audio_dir", type=str, default='./data/feats/vggish',
                        help="audio features dir")
    parser.add_argument("--video_dir", type=str, default='./data/feats/res152',
                        help="2D visual features dir")
    parser.add_argument("--st_dir", type=str, default='./data/feats/r2plus1d_18',
                        help="3D visual features dir")
    parser.add_argument("--v_pseudo_data_dir", type=str, default='./data/feats_CLIP_large/framewise_pseudo_labels',
                        help="visual segment-level pseudo labels dir")
    parser.add_argument("--a_pseudo_data_dir", type=str, default='./data/feats_CLAP_new/framewise_pseudo_labels',
                        help="audio segment-level pseudo labels dir")

    parser.add_argument("--label_train", type=str, default="./data/AVVP_train.csv",
                        help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="./data/AVVP_val_pd.csv",
                        help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="./data/AVVP_test_pd.csv",
                        help="weak test csv file")

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'val', 'test'],
                        help="which mode to use")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='the value for gradient clipping (0 means no gradient clipping)')

    # optimizer hyper-parameters
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # scheduler hyper-parameters
    parser.add_argument('--scheduler', type=str, default='steplr', help='which scheduler to use')
    parser.add_argument('--stepsize', type=int, default=10, help='step size of learning scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of learning scheduler')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='the number of epochs for warm up')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='the minimum lr for lr decay')

    # model hyper-parameters
    parser.add_argument("--model", type=str, default='MMIL_Net', help="which model to use")
    parser.add_argument("--input_v_dim", type=int, default=2048)
    parser.add_argument("--input_a_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--norm_where", type=str, default="post_norm", choices=['post_norm', 'pre_norm'])

    parser.add_argument("--model_name", type=str,
                        help="the name for the model")
    parser.add_argument("--model_save_dir", type=str, default='models/',
                        help="where to save the trained model")

    # wandb configurations
    parser.add_argument("--use_wandb", action="store_true",
                        help="use wandb or not")
    parser.add_argument("--wandb_project_name", type=str, default='Baseline')
    parser.add_argument("--wandb_run_name", type=str)


    args = parser.parse_args()

    if args.model_name == None:
        args.model_name = args.model
    if 'CLAP' in args.audio_dir:
        print('reset args.input_a_dim')
        args.input_a_dim = 768
    if 'CLIP' in args.video_dir:
        print('reset args.input_v_dim')
        args.input_v_dim = 768             # 1024 or 768 (before visual projection / after visual projection)
    print('args =', args)

    if args.mode == 'train':
        assert not os.path.exists(os.path.join(args.model_save_dir, args.model_name)), "{} already exists. Please specify another model_name.".format(args.model_name)

        os.mkdir(os.path.join(args.model_save_dir, args.model_name))
        args_dict = args.__dict__
        with open(os.path.join(args.model_save_dir, args.model_name, "arguments.txt"), 'w') as f:
            f.writelines('-------------------------start-------------------------\n')
            for key, value in args_dict.items():
                f.writelines(key + ': ' + str(value) + '\n')
            f.writelines('--------------------------end--------------------------\n')

    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name)
        if args.wandb_run_name != None:
            wandb.run.name = args.wandb_run_name
        wandb.config.update(args)

    # Set random seed and device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MMIL_Net(args).to(device)

    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                    'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                    'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                    'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                    'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                    'Clapping']

    if args.mode == 'train':

        train_dataset = LLP_dataset(mode=args.mode, label=args.label_train, audio_dir=args.audio_dir, res152_dir=args.video_dir,
                                    r2plus1d_18_dir=args.st_dir, v_pseudo_data_dir=args.v_pseudo_data_dir, a_pseudo_data_dir=args.a_pseudo_data_dir)
        val_dataset   = LLP_dataset(mode='val', label=args.label_val, audio_dir=args.audio_dir, res152_dir=args.video_dir,
                                    r2plus1d_18_dir=args.st_dir, v_pseudo_data_dir=args.v_pseudo_data_dir, a_pseudo_data_dir=args.a_pseudo_data_dir)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        # Create loss function(s)
        criterion = nn.BCELoss()

        # Create optimizer, scheduler
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

        if args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.scheduler == 'warm_up_cos_anneal':
            print('Using hand-made lr scheduler')
        else:
            print('Warning! Not using any lr scheduler!')


        best_F = {'Seg-type': 0.0, 'Seg-a': 0.0, 'Seg-v': 0.0}
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            if args.scheduler == 'warm_up_cos_anneal':
                lr_warm_up_cos_anneal(optimizer, epoch, args.warm_up_epoch, args.epochs, args.lr_min, args.lr)

            cur_lr = optimizer.param_groups[0]['lr']
            start_time = time.time()
            train_loss_dict = train(args, model, train_loader, optimizer, criterion, epoch, device)
            
            if args.scheduler != 'warm_up_cos_anneal':
                scheduler.step()

            F_scores, val_loss_dict = eval(args, model, val_loader, './data', criterion, device)
            elapse_time = time.time() - start_time
            
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.model_name, "checkpoint_epoch_{}.pt".format(epoch)))
            if F_scores['Seg-type'] > best_F['Seg-type']:
                best_F = F_scores
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.model_name, "checkpoint_best.pt"))
            
            if args.use_wandb:
                wandb.log({
                    'val loss_video': val_loss_dict['loss_video'],
                    'val loss_valor_v': val_loss_dict['loss_valor_v'],
                    'val loss_valor_a': val_loss_dict['loss_valor_a'],
                    'val loss_all': val_loss_dict['loss_all'],
                    'Segment-level audio F score': F_scores['Seg-a'],
                    'Segment-level visual F score': F_scores['Seg-v'],
                    'Segment-level type F score': F_scores['Seg-type'],
                    'lr': cur_lr,
                    'epoch': epoch
                })

            print('Epoch[{}/{}](Time:{:.2f} sec)(lr:{:.6f}) Train Loss: {:.3f} Val Loss: {:.3f}  Val F: {:.3f}'.format(
                    epoch, args.epochs, elapse_time, cur_lr, train_loss_dict['loss_all'], val_loss_dict['loss_all'], F_scores['Seg-type']))

        print('-'*30)
        print('Best F scores (at epoch {}):'.format(best_epoch))
        f_scores_tb = get_evaluation_result_table(mode="Val", F_scores_dict=best_F)
        print(f_scores_tb)

    elif args.mode == 'val':
        val_dataset = LLP_dataset(mode='val', label=args.label_val, audio_dir=args.audio_dir, res152_dir=args.video_dir,
                                    r2plus1d_18_dir=args.st_dir, v_pseudo_data_dir=args.v_pseudo_data_dir, a_pseudo_data_dir=args.a_pseudo_data_dir)
        val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.model_save_dir, args.model_name, "checkpoint_best.pt")))

        # Evaluation
        F_scores_val, _ = eval(args, model, val_loader, './data', criterion=None, device=device)
        f_scores_tb = get_evaluation_result_table(mode="Val", F_scores_dict=F_scores_val)
        print('Evaluation result:')
        print(f_scores_tb)

    elif args.mode == 'test':
        test_dataset = LLP_dataset(mode=args.mode, label=args.label_test, audio_dir=args.audio_dir, res152_dir=args.video_dir,
                                    r2plus1d_18_dir=args.st_dir, v_pseudo_data_dir=args.v_pseudo_data_dir, a_pseudo_data_dir=args.a_pseudo_data_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.model_save_dir, args.model_name, "checkpoint_best.pt")))

        # Evaluation
        F_scores_test, _ = eval(args, model, test_loader, './data', criterion=None, device=device)
        f_scores_tb = get_evaluation_result_table(mode="Test", F_scores_dict=F_scores_test)
        print('Evaluation result:')
        print(f_scores_tb)

    else:
        print('Please specify args.mode!')
        

if __name__ == '__main__':
    main()
