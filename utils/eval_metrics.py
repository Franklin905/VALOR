import os
import csv
import numpy as np
import pandas as pd

import torch


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(x)
    return p / N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p / N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p / N


def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # extract events
    N = 25
    event_p_a = [None for n in range(25)]
    event_gt_a = [None for n in range(25)]
    event_p_v = [None for n in range(25)]
    event_gt_v = [None for n in range(25)]
    event_p_av = [None for n in range(25)]
    event_gt_av = [None for n in range(25)]

    TP_a = np.zeros(25)
    TP_v = np.zeros(25)
    TP_av = np.zeros(25)

    FP_a = np.zeros(25)
    FP_v = np.zeros(25)
    FP_av = np.zeros(25)

    FN_a = np.zeros(25)
    FN_v = np.zeros(25)
    FN_av = np.zeros(25)

    for n in range(N):
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0  # all true negatives
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:
        f_v = 1.0  # all true negatives
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:
        f = 1.0  # all true negatives
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:
        f_av = 1.0  # all true negatives
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores = 2 * TP / (2 * TP + FP + FN)
    # all inputs shapes are (25, 10)
    # False negative: prediction shows negative, but it actually is positive
    # False positive: prediction shows positive, but it actually is negative
    TP_a = np.sum(SO_a * GT_a, axis=1)      # (25, )
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0  # all true negatives
    else:
        f_a = (sum(F_a) / len(F_a))     # average across classes

    if len(F_v) == 0:
        f_v = 1.0  # all true negatives
    else:
        f_v = (sum(F_v) / len(F_v))     # average across classes

    if len(F) == 0:
        f = 1.0  # all true negatives
    else:
        f = (sum(F) / len(F))           # average across classes
    if len(F_av) == 0:
        f_av = 1.0  # all true negatives
    else:
        f_av = (sum(F_av) / len(F_av))  # average across classes

    return f_a, f_v, f, f_av


def to_vec(start, end):
    x = np.zeros(10)
    for i in range(start, end):
        x[i] = 1
    return x


def extract_event(seq, n):
    x = []
    i = 0
    while i < 10:
        if seq[i] == 1:
            start = i
            if i + 1 == 10:
                i = i + 1
                end = i
                x.append(to_vec(start, end))
                break

            for j in range(i + 1, 10):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end))
                    break
                else:
                    i = j + 1
                    if i == 10:
                        end = i
                        x.append(to_vec(start, end))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # IoU, threshold=0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # 0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN


# Functions below are added by me.

def classwise_eval(model, data_loader, dataset_csv, gt_csv_dir, device):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(dataset_csv, header=0, sep='\t')
    df_a = pd.read_csv(os.path.join(gt_csv_dir, "AVVP_eval_audio.csv"), header=0, sep='\t')
    df_v = pd.read_csv(os.path.join(gt_csv_dir, "AVVP_eval_visual.csv"), header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = [[] for _ in range(len(categories))]
    F_seg_v = [[] for _ in range(len(categories))]
    F_seg = [[] for _ in range(len(categories))]
    F_seg_av = [[] for _ in range(len(categories))]

    audio_stat = {"TP": np.zeros(25, dtype=np.int32), "FP": np.zeros(25, dtype=np.int32), "FN": np.zeros(25, dtype=np.int32)}
    visual_stat = {"TP": np.zeros(25, dtype=np.int32), "FP": np.zeros(25, dtype=np.int32), "FN": np.zeros(25, dtype=np.int32)}
    av_stat = {"TP": np.zeros(25, dtype=np.int32), "FP": np.zeros(25, dtype=np.int32), "FN": np.zeros(25, dtype=np.int32)}

    # Record per video class-wise TP, FN, FP
    audio_per_video_classwise_stats = []
    visual_per_video_classwise_stats = []

    a_usually_full_video = np.zeros(25, dtype=np.int32)
    a_num_of_occurred_videos = np.zeros(25, dtype=np.int32)
    v_usually_full_video = np.zeros(25, dtype=np.int32)
    v_num_of_occurred_videos = np.zeros(25, dtype=np.int32)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            video_name = batch_data['video_name']
            # batch_data = {key: value.to(device) for key, value in batch_data.items() if key != 'video_name' or key != 'idx'}
            video_res_feats, video_3d_feats, audios = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device)

            output, _, _, frame_prob, frame_logits, _, _ = model(audios, video_res_feats, video_3d_feats)  # (B, 25), (B, 25), (B, 25), (B, 10, 2, 25)

            # When video-level gt labels are used!!! --------------------------------------------
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)  # (B, 25)

            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()  # (10, 25)
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()  # (10, 25)

            # # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)     # (10, 25)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)     # (10, 25)
            # -----------------------------------------------------------------------------------

            # When video-level gt labels are NOT used ------------------------------------------
            # Pa = torch.sigmoid(frame_logits[0, :, 0, :]).cpu().detach().numpy()  # (10, 25)
            # Pv = torch.sigmoid(frame_logits[0, :, 1, :]).cpu().detach().numpy()  # (10, 25)

            # Pa = (Pa >= 0.5).astype(np.int_)            # (10, 25)
            # Pv = (Pv >= 0.5).astype(np.int_)            # (10, 25)
            # ----------------------------------------------------------------------------------

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

            GT_av = GT_a * GT_v     # (25, 10), (25, 10), (25, 10), all in one-hot labels

            # ===============================
            # per_class_a_seg_num = GT_a.sum(axis=1)
            # per_class_a_full_video = (per_class_a_seg_num >= 8).astype(np.int32)
            # per_class_a_occurred_video = (per_class_a_seg_num >= 1).astype(np.int32)

            # a_usually_full_video = a_usually_full_video + per_class_a_full_video
            # a_num_of_occurred_videos = a_num_of_occurred_videos + per_class_a_occurred_video

            # per_class_v_seg_num = GT_v.sum(axis=1)
            # per_class_v_full_video = (per_class_v_seg_num >= 9).astype(np.int32)
            # per_class_v_occurred_video = (per_class_v_seg_num >= 1).astype(np.int32)

            # v_usually_full_video = v_usually_full_video + per_class_v_full_video
            # v_num_of_occurred_videos = v_num_of_occurred_videos + per_class_v_occurred_video
            # ===============================



            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v         # (25, 10)

            # segment-level F1 scores
            F_a, F_v, F, F_av = classwise_segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)            
            for i in range(len(categories)):
                if len(F_a[i]) != 0:
                    F_seg_a[i].append(F_a[i][0])
                if len(F_v[i]) != 0:
                    F_seg_v[i].append(F_v[i][0])
                if len(F[i]) != 0:
                    F_seg[i].append(F[i][0])
                if len(F_av[i]) != 0:
                    F_seg_av[i].append(F_av[i][0])

            # segment-level TP, FP, FN
            TP_a, FN_a, FP_a, TP_v, FN_v, FP_v, TP_av, FN_av, FP_av = classwise_segment_level_2(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            audio_stat['TP'] += TP_a; audio_stat['FN'] += FN_a; audio_stat['FP'] += FP_a
            visual_stat['TP'] += TP_v; visual_stat['FN'] += FN_v; visual_stat['FP'] += FP_v
            av_stat['TP'] += TP_av; av_stat['FN'] += FN_av; av_stat['FP'] += FP_av

            # record class-wise TP, FN, FP of a video (TODO: choose to write audio (f_a) or visual (f_v))
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            visual_per_video_classwise_stats.append([video_name[0]] + ["{:.4f}".format(f_v)])
            visual_per_video_classwise_stats.append(['', 'TP'] + [int(k) for k in TP_v])
            visual_per_video_classwise_stats.append(['', 'FN'] + [int(k) for k in FN_v])
            visual_per_video_classwise_stats.append(['', 'FP'] + [int(k) for k in FP_v])
            audio_per_video_classwise_stats.append([video_name[0]] + ["{:.4f}".format(f_a)])
            audio_per_video_classwise_stats.append(['', 'TP'] + [int(k) for k in TP_a])
            audio_per_video_classwise_stats.append(['', 'FN'] + [int(k) for k in FN_a])
            audio_per_video_classwise_stats.append(['', 'FP'] + [int(k) for k in FP_a])

    # print('a_usually_full_video =', a_usually_full_video)
    # print('a_num_of_occurred_videos =', a_num_of_occurred_videos)

    for i in range(len(categories)):
        F_seg_a[i] = 100 * np.mean(np.array(F_seg_a[i]))
        F_seg_v[i] = 100 * np.mean(np.array(F_seg_v[i]))
        F_seg[i] = 100 * np.mean(np.array(F_seg[i]))
        F_seg_av[i] = 100 * np.mean(np.array(F_seg_av[i]))
    
    return F_seg_a, F_seg_v, F_seg_av, F_seg, audio_stat, visual_stat, av_stat, audio_per_video_classwise_stats, visual_per_video_classwise_stats

def classwise_segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores = 2 * TP / (2 * TP + FP + FN)
    # all inputs shapes are (25, 10)
    # False negative: prediction shows negative, but the truth is positive
    # False positive: prediction shows positive, but the truth is negative
    TP_a = np.sum(SO_a * GT_a, axis=1)      # (25, )
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)

    n = len(FP_a)
    F_a = [[] for _ in range(n)]
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a[ii].append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = [[] for _ in range(n)]
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v[ii].append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = [[] for _ in range(n)]
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F[ii].append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = [[] for _ in range(n)]
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av[ii].append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    return F_a, F_v, F, F_av    # lists of lists

def classwise_segment_level_2(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores = 2 * TP / (2 * TP + FP + FN)
    # all inputs shapes are (25, 10)
    # False negative: prediction shows negative, but the truth is positive
    # False positive: prediction shows positive, but the truth is negative
    TP_a = np.sum(SO_a * GT_a, axis=1).astype(np.int16)      # (25, )
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1).astype(np.int16)
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1).astype(np.int16)

    TP_v = np.sum(SO_v * GT_v, axis=1).astype(np.int16)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1).astype(np.int16)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1).astype(np.int16)

    TP_av = np.sum(SO_av * GT_av, axis=1).astype(np.int16)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1).astype(np.int16)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1).astype(np.int16)

    return TP_a, FN_a, FP_a, TP_v, FN_v, FP_v, TP_av, FN_av, FP_av


def get_pred_frames(cur_video_name, preds, categories):
    """
    Arguments:
        cur_video_name (str): the video name of the current data
        preds (ndarray): (25, 10), a frame-level one-hot prediction of a video
        categories (list): the list consisting of all category names
    Return:
        records (list): a list containing predicted start/end frames of each event in a video
    """

    records = []
    event_exist_indices = np.where(np.sum(preds, axis=1))[0]
    for event_idx in event_exist_indices:
        start_frame = 0
        end_frame = -1
        for i in range(10):
            if preds[event_idx][i] == 0:
                if end_frame == -1:
                    start_frame = i+1
                else:
                    records.append([cur_video_name, start_frame, end_frame+1, categories[event_idx]])
                    start_frame = i+1
                    end_frame = -1
            else:
                if end_frame == -1:
                    end_frame = i
                else:
                    end_frame += 1
        
        if end_frame != -1:
            records.append([cur_video_name, start_frame, end_frame+1, categories[event_idx]])

    return records


def calculate_video_level_F_score(args, model, data_loader, device):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    F_dict = {'F score': 0, 'total_data': 0}

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # video_name = batch_data['video_name']
            # batch_data = {key: value.to(device) for key, value in batch_data.items() if key != 'video_name' or key != 'idx'}
            video_res_feats, video_3d_feats, audios, labels = batch_data['video_s'].to(device), batch_data['video_st'].to(device), batch_data['audio'].to(device), batch_data['label'].float().to(device)

            output, a_prob, v_prob, frame_prob, frame_logits, _, _ = model(audios, video_res_feats, video_3d_feats)  # (B, 25)
            video_level_preds = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)  # (B, 25)

            f_scores = calculate_per_video_video_level_F_score(video_level_preds, labels.cpu().detach().numpy().astype(np.int_))
            F_dict['F score'] += f_scores.sum()
            F_dict['total_data'] += len(f_scores)

    return (F_dict['F score'] / F_dict['total_data'])

def calculate_per_video_video_level_F_score(preds, gts):
    # preds shape = (B, 25), gts shape = (B, 25), both are ndarray
    # F score = 2 * TP / (2 * TP + FP + FN)

    TP = np.sum(preds * gts, axis=1)            # (B, )
    FN = np.sum((1 - preds) * gts, axis=1)      # (B, )
    FP = np.sum(preds * (1 - gts), axis=1)      # (B, )
    
    return (2 * TP / (2 * TP + FP + FN))