import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, hidden_dim):
        super(Encoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm_a = nn.LayerNorm(hidden_dim)
        self.final_norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, norm_where, src_a, src_v, mask=None, src_key_padding_mask=None):

        for i in range(self.num_layers):
            src_a = self.layers[i](norm_where, src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)
            src_v = self.layers[i](norm_where, src_v, src_a, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)

        if norm_where == "pre_norm":
            src_a = self.final_norm_a(src_a)
            src_v = self.final_norm_v(src_v)

        return src_a, src_v


class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, norm_where, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if norm_where == "post_norm":
            if with_ca:
                src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
                src_q = self.norm1(src_q)
            else:
                src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)
                src_q = self.norm1(src_q)

            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
            src_q = src_q + self.dropout2(src2)
            src_q = self.norm2(src_q)

            return src_q.permute(1, 0, 2)
        
        elif norm_where == "pre_norm":
            src_q_pre_norm = self.norm1(src_q)

            if with_ca:
                src1 = self.cm_attn(src_q_pre_norm, src_v, src_v, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            else:
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)

            src_q_pre_norm = self.norm2(src_q)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q_pre_norm))))
            src_q = src_q + self.dropout2(src2)

            return src_q.permute(1, 0, 2)
        
        else:
            raise ValueError('norm_where should be pre_norm or post_norm')


class MMIL_Net(nn.Module):

    def __init__(self, args):
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(args.hidden_dim, 25)
        self.fc_frame_att = nn.Linear(args.hidden_dim, 25)
        self.fc_av_att = nn.Linear(args.hidden_dim, 25)

        self.fc_a =  nn.Linear(args.input_a_dim, args.hidden_dim)

        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_st = nn.Linear(512, args.hidden_dim)
        self.fc_fusion = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.hat_encoder = Encoder(HANLayer(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim),
                                   num_layers=args.num_layers,
                                   hidden_dim=args.hidden_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim     # 2048: ResNet152, 768: CLIP large
        self.input_a_dim = args.input_a_dim     # 128: VGGish, 512: CLAP
        self.hidden_dim = args.hidden_dim

    def forward(self, audio, visual, visual_st):

        if audio.size(1) == 64:     # input data are feature maps
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, self.input_a_dim, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, self.hidden_dim, 10)
            x1 = x1.permute(0, 2, 1)
        else:
            x1 = self.fc_a(audio)


        # 2d and 3d visual feature fusion
        if visual.size(1) == 80:        # input 2d features are from ResNet152
            vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
            vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        else:                           # input 2d features are from CLIP
            vid_s = self.fc_v(visual)


        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)

        # HAN
        x1, x2 = self.hat_encoder(self.norm_where, x1, x2)

        # prediction
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
        frame_logits = self.fc_prob(x)                                  # (B, T, 2, C)
        frame_prob = torch.sigmoid(frame_logits)                        # (B, T, 2, C)

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)          # (B, T, 2, C)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)                # (B, T, 2, C)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)      # (B, C)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)       # (B, C)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)       # (B, C)

        return global_prob, a_prob, v_prob, frame_prob, frame_logits