import os
import argparse
import pandas as pd

def download(videos_saved_dir, name, t_seg):

    if not os.path.exists(videos_saved_dir):
        os.makedirs(videos_saved_dir)

    link_prefix = "https://www.youtube.com/watch?v="

    filename_full_video = os.path.join(videos_saved_dir, name) + "_full_video.mp4"
    filename = os.path.join(videos_saved_dir, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print("{} already exists, skip".format(filename))
        return

    print( "download the whole video for: [%s] - [%s]" % (videos_saved_dir, name))
    command1 = 'yt-dlp ' + link + ' -o ' + filename_full_video + ' -f mp4 -q'

    os.system(command1)


    t_start, t_end = t_seg
    t_dur = t_end - t_start
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    command2 = 'ffmpeg '
    command2 += '-ss '
    command2 += str(t_start) + ' '
    command2 += '-i '
    command2 += filename_full_video + ' '
    command2 += '-t '
    command2 += str(t_dur) + ' '
    command2 += '-vcodec libx264 '
    command2 += '-acodec aac -strict -2 '
    command2 += filename + ' '
    command2 += '-y '           # overwrite without asking
    command2 += '-loglevel -8 ' # print no log
    os.system(command2)
    try:
        os.remove(filename_full_video)
    except:
        return

    print ("finish the video as: " + filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_all_dataset", type=str, default="./data/AVVP_dataset_full.csv")
    parser.add_argument("--videos_saved_dir", type=str, default='/home/yuhsuanchen/Desktop/mnt/AVVP_Dataset/videos_download')

    args = parser.parse_args()

    # %% read the video trim time indices
    filename_source = args.label_all_dataset
    df = pd.read_csv(filename_source, header=0, sep='\t')
    filenames = df["filename"]
    length = len(filenames)
    print('# of videos to be downloaded: {}'.format(length))

    names = []
    segments = {}
    print('Videos will be saved in {}'.format(args.videos_saved_dir))
    for i in range(length):
        row = df.loc[i, :]
        name = row[0][:11]
        steps = row[0][11:].split("_")
        t_start = float(steps[1])
        t_end = t_start + 10
        segments[name] = (t_start, t_end)
        download(args.videos_saved_dir, name, segments[name])
        names.append(name)

    print('# of downloaded videos: {}'.format(len(os.listdir(args.videos_saved_dir))))
