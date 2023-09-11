import os
import csv
import librosa
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import laion_clap
from clap_utils import My_CLAP, get_audio_features, int16_to_float32, float32_to_int16

import warnings
warnings.filterwarnings("ignore")

def ids_to_multinomial(ids):
    """
    Multi-label one-hot label encoding

    Argument
        ids (list): a list of class indices
    Output
        y (ndarray): (num of classes, ), multi-label one-hot labels, e.g. [1,0,1,0,0,...]
    """

    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y


class LLP_dataset(Dataset):

    def __init__(self, label_csv):
        self.df = pd.read_csv(label_csv, header=0, sep='\t')
        self.filenames = self.df["filename"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        sample = {'label': label, 'video_name': row[0]}

        return sample


def get_text_embedding(model, event_captions):
    """
    Extract event embeddings from input event captions with the prompt added

    Output
        text_embeds (ndarray): shape = (25, 512), L2-normalized
    """

    with torch.no_grad():
        text_embeds = model.get_text_embedding(event_captions)

        return text_embeds


def get_audio_embedding(model, video_name, audio_path):
    """
    Extract segment-level audio embeddings

    Output
        audio_embeds (ndarray): shape = (10, 512), L2-normalized
    """
    
    with torch.no_grad():

        path_to_audio = os.path.join(audio_path, video_name[:11]+'.mp4')     # ".mp4" or ".wav"
        audio_waveform, sr = librosa.load(path_to_audio, sr=48000)

        # Split audio data into segments
        samples = np.round(np.linspace(0, audio_waveform.shape[0], 11)).astype(np.int32)
        step = audio_waveform.shape[0] // 10
        seg_boundaries = []
        for i in range(len(samples)-1):
            seg_boundaries.append([samples[i], samples[i]+step])  # [start point, end point]

        framewise_audio = [audio_waveform[seg_boundaries[i][0]: seg_boundaries[i][1]] for i in range(len(seg_boundaries))]
        assert len(framewise_audio) == 10

        audio_embeds = model.get_audio_embedding_from_data(x = framewise_audio)

        return audio_embeds


def calculate_audio_logits(model, audio_embeds, text_embeds, device):
    """
    Calculate segment-wise event logits

    Output
        logits_audio_text (tensor): size = (10, 25), scaled by logit_scale_a
    """

    with torch.no_grad():
        logit_scale_a, logit_scale_t = model.model(None, None, device)
        logit_scale_a = logit_scale_a.to(device)

        # Calculate similarities
        logits_audio_text = logit_scale_a * audio_embeds @ text_embeds.t()

        return logits_audio_text
    

def get_segment_pseudo_labels(logits_audio_text, thresholds, labels, device):
    """
    Construct segment-level audio pseudo labels (multi-label one-hot pseudo labels)
    based on the pre-defined thresholds, logits, and the video-level labels

    Output
        Pa (tensor): size = (10, 25), segment-level audio pseudo labels
    """

    with torch.no_grad():
        occurred_event_idx = labels.nonzero().squeeze(dim=-1)
        
        occurred_events_logits = logits_audio_text[:, occurred_event_idx]
        class_thresholds = thresholds[:, occurred_event_idx]
        segment_preds = torch.where(occurred_events_logits > class_thresholds, 1, 0).long()

        Pa = torch.zeros(10, 25).long().to(device)
        Pa[:, occurred_event_idx.long()] = segment_preds

    return Pa


def audio_label_elaboration(model, data_loader, event_captions, audio_path, thresholds, save_path, device, print_progress):
    '''
    Audio label elaboration in VALOR

    Arguments
        model: a pre-trained CLAP model
        data_loader (DataLoader): a data loader for training, validation, or testing split
        event_captions (list): a list containing all event captions (with the prompt added) in the same order as in categories
        audio_path (str): the directory where raw audio data (mp4 or wav files) are saved
        thresholds (list): a list containing the threshold for each event
        save_path (str): the directory where the pseudo labels are going to be saved
        device (torch.device): gpu/cpu
    '''

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))

    text_embeds = get_text_embedding(model, event_captions)    # ndarray, shape = (25, 512), L2-normalized already
    text_embeds = torch.from_numpy(text_embeds).to(device)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            labels, video_name = batch_data['label'].to(device).squeeze(), batch_data['video_name'][0]

            audio_embeds = get_audio_embedding(model, video_name, audio_path)     # ndarray, shape = (10, 512), L2-normalized already
            audio_embeds = torch.from_numpy(audio_embeds).to(device)
            logits_audio_text = calculate_audio_logits(model, audio_embeds, text_embeds, device)
            seg_pseudo_labels = get_segment_pseudo_labels(logits_audio_text, thresholds, labels, device)

            np.save(os.path.join(save_path, video_name[:11]+'.npy'), seg_pseudo_labels.cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
        print('')

    return


def get_audio_feature_maps(my_model, data_loader, audio_path, save_path, print_progress):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            video_name = batch_data['video_name'][0]
            path_to_audio = os.path.join(audio_path, video_name[:11]+'.mp4')     # ".mp4" if load from video, else ".wav"
            audio_waveform, sr = librosa.load(path_to_audio, sr=48000)

            # Make sure all audio lengths are 10 seconds
            samples = np.round(np.linspace(0, len(audio_waveform) - 1, 480000)).astype(np.int32)
            audio_waveform = audio_waveform[samples]

            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000,
                data_truncating='fusion', 
                data_filling='repeatpad',
                audio_cfg=my_model.clap_cfg['audio_cfg']
            )

            # Extract segment-wise audio feature maps
            audio_feat_maps = my_model.get_audio_feature_maps([temp_dict])  # (B, 64, 768)

            np.save(os.path.join(save_path, video_name[:11] + '.npy'), audio_feat_maps.squeeze(0).cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
        print('')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='harvesting segment-level audio pseudo labels with CLAP')

    parser.add_argument("--audio_dir", type=str, default='./data/raw_videos', help="the directory where .mp4 or .wav files are saved")
    parser.add_argument("--label_all_dataset", type=str, default="./data/AVVP_dataset_full.csv")

    parser.add_argument("--pseudo_labels_saved_dir", type=str, default='./data/CLAP/segment_pseudo_labels')
    parser.add_argument("--audio_feats_saved_dir", type=str, default='./data/CLAP/features')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument('--print_progress', action='store_true')

    args = parser.parse_args()


    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    print('===> Prepare model ...')
    model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
    model.load_ckpt(model_id=2)     # 0: 630k non-fusion; 1: 630k+audioset non-fusion; 2: 630k fusion; 3: 630k+audioset fusion
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print('===> Prepare dataloader ...')
    whole_dataset = LLP_dataset(label_csv=args.label_all_dataset)
    whole_loader  = DataLoader(whole_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    '''
    generate audio pseudo labels
    '''
    event_captions = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying (food)',
                    'Basketball bounce', 'Fire alarm', 'Chainsaw', 'Cello', 'Banjo',
                    'Singing', 'Chicken, rooster', 'Violin fiddle', 'Vacuum cleaner',
                    'Baby laughter', 'Accordion', 'Lawn mower', 'Motorcycle', 'Helicopter',
                    'Acoustic guitar', 'Telephone bell ringing', 'Baby cry, infant cry', 'Blender',
                    'Clapping']
    event_captions = ["This is a sound of " + t.lower() for t in event_captions]

    thresholds = np.array([0, 0, 1, 4, 6, -2, 4, 4, 2, 2,
                            2, 1, 2, 3, 0, 2, 2, 2, 0, 2,
                            -1, 2, 3, 3, 0])
    thresholds = torch.from_numpy(thresholds).to(device).unsqueeze(0).expand(10, -1)

    print('===> Generate audio pseudo labels ...')
    print('(labels will be saved at {})'.format(args.pseudo_labels_saved_dir))
    audio_label_elaboration(model, whole_loader, event_captions, args.audio_dir, thresholds,
                            args.pseudo_labels_saved_dir, device, args.print_progress)
    print()
    

    '''
    extract audio feature maps
    '''
    my_model = My_CLAP(model, device)
    my_model.eval()
    for param in my_model.parameters():
        param.requires_grad = False

    print('===> Generate audio segment features ...')
    print('(features will be saved at {})'.format(args.audio_feats_saved_dir))
    get_audio_feature_maps(my_model, whole_loader, args.audio_dir,
                            args.audio_feats_saved_dir, args.print_progress)


