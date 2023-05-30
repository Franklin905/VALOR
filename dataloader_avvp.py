import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


def ids_to_multinomial(ids):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
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

    def __init__(self, mode, label, audio_dir, res152_dir, r2plus1d_18_dir,
                 v_pseudo_data_dir, a_pseudo_data_dir):
        self.mode = mode
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = res152_dir
        self.st_dir = r2plus1d_18_dir
        self.num_of_data = len(self.filenames)

        self.v_pseudo_data_dir = v_pseudo_data_dir
        self.a_pseudo_data_dir = a_pseudo_data_dir


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label, 'video_name': row[0]}

        sample['audio_pseudo_labels'] = np.load(os.path.join(self.a_pseudo_data_dir, name + '.npy'))
        sample['visual_pseudo_labels'] = np.load(os.path.join(self.v_pseudo_data_dir, name + '.npy'))

        return sample