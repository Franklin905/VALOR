# Video Issues

When using `download_dataset.py` to obtain the original videos, there are some issues:
1. Some videos have been taken down from YouTube, so they cannot be downloaded.

2. [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20) authors originally used the "[youtube-dl](https://github.com/ytdl-org/youtube-dl)" package for video download. Our features / labels are extracted from this original version of videos. However, their downloading script does not work for me as of 08/29/2023. As a result, I switched to using the "[yt-dlp](https://github.com/yt-dlp/yt-dlp)" package to provide a reference downloading script. Compared to the first version, this newer version of videos have minor distinctions that are imperceptible to the eyes, leading to slight differences in the extracted features compared to the features downloaded from the Google Drive link. Fortunately, the pseudo labels generated from these two different versions of videos seem mostly the same. (We examined 5 videos, and the audio and visual pseudo labels generated from these two different versions of videos are exactly identical.)

For best reproducibility down to the bits and bytes, we recommend using our features / labels from the Google Drive. The feature extraction / pseudo-label generation code are released as a reference for easier extension to more pre-trained models of the users' choice. Unfortunately, we cannot share the original videos due to copyright and legal concerns.
