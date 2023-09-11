import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

'''
Direct copy from official CLAP GitHub
'''
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    )(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=64,
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)

def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    """
    with torch.no_grad():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample


class My_CLAP(nn.Module):

    def __init__(self, clap, device):
        super(My_CLAP, self).__init__()
        self.device = device
        self.clap = clap.model
        self.clap.eval()
        for param in self.clap.parameters():
            param.requires_grad = False

        self.clap_cfg = clap.model_cfg

    def get_audio_feature_maps(self, data):

        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(self.device)
        
        output_dict = self.new_encode_audio(input_dict, device=self.device)
        output_embeds = output_dict["feat_maps"]
        
        return output_embeds

    # re-write original CLAP function
    def new_encode_audio(self, audio, device):
        return self.htsat_forward(audio, mixup_lambda=None, device=device)
    
    # re-write original CLAP function
    def get_audio_embedding(self, data):

        device = next(self.clap.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        
        output_dict = self.new_encode_audio(input_dict, device=device)
        output_embed = output_dict["embedding"]

        audio_embeds = self.clap.audio_projection(output_embed)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        
        return audio_embeds
    
    # re-write original CLAP's HTS-AT fucntion
    def htsat_forward(self, x: torch.Tensor, mixup_lambda = None, device=None):

        if self.clap.audio_branch.enable_fusion and x["longer"].sum() == 0:

            x = x["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.clap.audio_branch.bn0(x)
            x = x.transpose(1, 3)
            x = self.clap.audio_branch.reshape_wav2img(x)
            output_dict = self.htsat_forward_features(x, longer_idx=[])

            return output_dict
    
    # re-write original CLAP's HTS-AT fucntion
    def htsat_forward_features(self, x, longer_idx = None):
        # A deprecated optimization for using a hierarchical output from different blocks
        # x size = (B, 4, 256, 256)

        frames_num = x.shape[2]
        x = self.clap.audio_branch.patch_embed(x, longer_idx = longer_idx)

        if self.clap.audio_branch.ape:
            x = x + self.clap.audio_branch.absolute_pos_embed
        x = self.clap.audio_branch.pos_drop(x)
        for i, layer in enumerate(self.clap.audio_branch.layers):
            x, attn = layer(x)
        # for x
        x = self.clap.audio_branch.norm(x)      # (B, 64, 768)

        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.clap.audio_branch.depths) - 1)) // self.clap.audio_branch.patch_stride[0]
        ST = frames_num // (2 ** (len(self.clap.audio_branch.depths) - 1)) // self.clap.audio_branch.patch_stride[1]
        x = x.permute(0,2,1).contiguous().reshape(B, C, SF, ST)     # (B, 768, 8, 8)
        B, C, F, T = x.shape
        # group 2D CNN
        c_freq_bin = F // self.clap.audio_branch.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)                     # (B, 768, 4, 2, 8)
        x = x.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)     # (B, 768, 2, 32)

        # get latent_output
        fine_grained_latent_output = torch.mean(x, dim = 2)
        fine_grained_latent_output = self.interpolate(fine_grained_latent_output.permute(0,2,1).contiguous(), 8 * self.clap.audio_branch.patch_stride[1]) 
        
        temp = torch.flatten(x, 2)          # (B, 768, 64)
        latent_output = self.clap.audio_branch.avgpool(temp)    # (B, 768, 1)
        latent_output = torch.flatten(latent_output, 1)         # (B, 768)

        # display the attention map, if needed

        x = self.clap.audio_branch.tscam_conv(x)
        x = torch.flatten(x, 2) # B, C, T
 
        fpx = self.interpolate(torch.sigmoid(x).permute(0,2,1).contiguous(), 8 * self.clap.audio_branch.patch_stride[1]) 
            
        x = self.clap.audio_branch.avgpool(x)
        x = torch.flatten(x, 1)

        output_dict = {
            'framewise_output': fpx, # already sigmoided
            'clipwise_output': torch.sigmoid(x),
            'fine_grained_embedding': fine_grained_latent_output,   # (B, 1024, 768)
            'embedding': latent_output,                             # (B, 768)
            'feat_maps': temp.permute(0, 2, 1).contiguous()         # (B, 64, 768)
        }

        return output_dict
    
    def interpolate(self, x, ratio):
        """Interpolate data in time domain. This is used to compensate the
        resolution reduction in downsampling of a CNN.
        Args:
        x: (batch_size, time_steps, classes_num)
        ratio: int, ratio to interpolate
        Returns:
        upsampled: (batch_size, time_steps * ratio, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
        return upsampled