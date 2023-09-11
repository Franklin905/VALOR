import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# huggingface
from transformers import CLIPProcessor, CLIPModel

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

    def __init__(self, label_csv, video_frame_dir, clip_processor):
        self.df = pd.read_csv(label_csv, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.video_frame_dir = video_frame_dir
        self.img_size = 224
        self.img_mean = clip_processor.image_processor.image_mean
        self.img_std = clip_processor.image_processor.image_std

        self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((self.img_size, self.img_size)),
                                                transforms.Normalize(self.img_mean, self.img_std)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        video_name = row[0][:11]

        image_list = sorted(glob.glob(os.path.join(self.video_frame_dir, video_name, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, 10))     # get 10 indices equally spaced

        image_list = [image_list[int(sample)] for sample in samples]
        video_frames = []
        for iImg in range(len(image_list)):
            img = Image.open(image_list[iImg]).convert('RGB')       # (H, W, C)
            video_frames.append(self.transform(img))
        video_frames = torch.stack(video_frames)

        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        sample = {'label': label, 'video_name': row[0], 'video_frames': video_frames}

        return sample
    

def calculate_visual_logits(model, video_frames, event_captions, device):
    """
    Calculate the event logits of each image from the constructed event captions and raw images

    Arguments
        model (nn.Module): a pre-trained CLIP
        video_frames (tensor): size = (10, 3, H, W), video frames (1fps)
        event_captions (list): a list containing event captions in the same order as categories
        device (torch.device): gpu/cpu

    Output
        logits_image_text (tensor): size = (10, 25)
    """

    with torch.no_grad():

        inputs_img = {key: value for key, value in event_captions.items()}
        inputs_img['pixel_values'] = video_frames.to(device)

        outputs = model(**inputs_img)
        logits_image_text = outputs.logits_per_image

        return logits_image_text
    

def get_segment_pseudo_labels(logits_image_text, thresholds, labels, device):
    """
    Construct segment-level visual pseudo labels (multi-label one-hot pseudo labels)
    based on the pre-defined thresholds, logits, and the video-level labels

    Output
        Pv (tensor): size = (10, 25), segment-level visual pseudo labels
    """

    with torch.no_grad():
        occurred_event_idx = labels.nonzero().squeeze(dim=-1)

        occurred_events_logits = logits_image_text[:, occurred_event_idx]
        class_thresholds = thresholds[:, occurred_event_idx]
        segment_preds = torch.where(occurred_events_logits > class_thresholds, 1, 0).long()     # (10, k) in one-hot labels

        Pv = torch.zeros(10, 25).long().to(device)
        Pv[:, occurred_event_idx.long()] = segment_preds

    return Pv


def visual_label_elaboration(model, clip_processor, data_loader, event_captions, thresholds, save_path, device, print_progress):
    '''
    Visual label elaboration in VALOR

    Arguments
        model (nn.Module): a pre-trained CLAP model
        clip_processor:
        data_loader (DataLoader): a data loader for training, validation, or testing split
        event_captions (list): a list containing all event captions (with the prompt added) in the same order as in categories
        thresholds (list): a list containing the threshold for each event
        save_path (str): the directory where the pseudo labels are going to be saved
        device (torch.device): gpu/cpu
    '''

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))

    event_captions = clip_processor(text=event_captions, images=None, return_tensors="pt", padding=True)
    event_captions = {key: value.to(device) for key, value in event_captions.items()}

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            labels, video_frames = batch_data['label'].to(device).squeeze(), batch_data['video_frames'].to(device).squeeze()
            video_name = batch_data['video_name'][0]
            logits_image_text = calculate_visual_logits(model, video_frames, event_captions, device)
            seg_pseudo_labels = get_segment_pseudo_labels(logits_image_text, thresholds, labels, device)

            np.save(os.path.join(save_path, video_name[:11]+'.npy'), seg_pseudo_labels.cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
        print('')

    return


def get_image_embedding(model, data_loader, save_path, device, print_progress):
    """
    Extract image embeddings from visual frames

    Arguments:
        model: a pre-trained CLIP
        data_loader (DataLoader):
        save_path (str): the directory where image features are going to be saved
        device (torch.device): gpu/cpu
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            video_frames = batch_data['video_frames'].to(device).squeeze()
            video_name = batch_data['video_name'][0]

            inputs_img = {'pixel_values': video_frames.to(device)}
            image_features = model.get_image_features(**inputs_img)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            np.save(os.path.join(save_path, video_name[:11]+'.npy'), image_features.cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
        print('')


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='harvesting segment-level visual pseudo labels with CLIP')

    parser.add_argument("--video_frame_dir", type=str, default='./data/video_frames')
    parser.add_argument("--label_all_dataset", type=str, default="./data/AVVP_dataset_full.csv")

    parser.add_argument("--pseudo_labels_saved_dir", type=str, default='./data/CLIP/segment_pseudo_labels')
    parser.add_argument("--visual_feats_saved_dir", type=str, default='./data/CLIP/features')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument('--print_progress', action='store_true')

    args = parser.parse_args()


    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    print('===> Prepare model ...')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print('===> Prepare dataloader ...')
    whole_dataset = LLP_dataset(label_csv=args.label_all_dataset, video_frame_dir=args.video_frame_dir, clip_processor=processor)
    whole_loader  = DataLoader(whole_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    event_captions = ['A photo of people talking.', 'A photo of a car.', 'A photo of people cheering.', 'A photo of a dog.', 'A photo of a cat.', 'A photo of frying food.',
                    'A photo of people playing basketball.', 'A photo of a fire alarm.', 'A photo of a chainsaw.', 'A photo of a cello.', 'A photo of a banjo.',
                    'A photo of people singing.', 'A photo of a chicken or a rooster.', 'A photo of a violin.', 'A photo of a vaccum cleaner.',
                    'A photo of a laughing baby.', 'A photo of an accordion.', 'A photo of a lawnmower.', 'A photo of a motorcycle.', 'A photo of a helicopter.',
                    'A photo of a acoustic guiter.', 'A photo of a ringing telephone.', 'A photo of a crying baby.', 'A photo of a blender.', 'A photo of hands clapping.']
    
    thresholds = np.array([20, 15, 18, 14, 15, 18, 18, 15, 15, 15,
                            15, 18, 15, 15, 15, 15, 15, 15, 15, 16,
                            14, 15, 15, 15, 18])
    thresholds = torch.from_numpy(thresholds).to(device).unsqueeze(0).expand(10, -1)

    print('===> Generate visual pseudo labels ...')
    print('(labels will be saved at {})'.format(args.pseudo_labels_saved_dir))
    visual_label_elaboration(model, processor, whole_loader, event_captions, thresholds,
                            args.pseudo_labels_saved_dir, device, args.print_progress)
    print()


    '''
    Extract visual embedding from each segment
    '''
    print('===> Generate visual segment features ...')
    print('(features will be saved at {})'.format(args.visual_feats_saved_dir))
    get_image_embedding(model, whole_loader, args.visual_feats_saved_dir,
                        device, args.print_progress)
