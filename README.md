![header](https://capsule-render.vercel.app/api?type=transparent&height=200&section=header&text=VMT%20for%20SUBS&fontSize=80&fontColor=345F53)

# CREATING DATASET 
### *prerequisites*
```
pip install moviepy
pip install imageio_ffmpeg
pip install webvtt-py
```
### 0. Clone the Repo
```
! git clone https://github.com/hyunbinui/VMT-for-SUBS.git
cd data
```
### 1. Downloading Youtube Videos & Subtitles
- create 'original_video' & 'original_subs' directory inside 'data' directory for Youtube videos and subtitles
```
mkdir original_video
mkdir original_subs
```
- (recommendation) get video ids from playlist
```
# playlist → youtube ids → txt file
youtube-dl --get-id [playlist link] -i >> list.txt
```
- download videos / subtitles (en-ko) from youtube by using [youtube-dl](https://github.com/ytdl-org/youtube-dl)

```
youtube-dl -a list.txt -o '/target_directory/original_video/%(id)s.%(ext)s' --rm-cache-dir --write-srt --sub-lang en,ko -o '/target_directory/original_subs/%(id)s.%(ext)s'
```

- if youtube-dl is way too slow, try using [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading videos

```
yt-dlp -a list.txt -o '/target_directory/original_video/%(id)s.%(ext)s' -S ext:mp4:m4a -i
youtube-dl -a list.txt --write-srt --sub-lang en,ko -o '/target_directory/original_subs/%(id)s.%(ext)s' --skip-download -i 
```
<br>

### 2. Constructing Dataset
- construct the text pair and video dataset by running the create_dataset.py file in 'data' directory
```
python create_dataset.py --idpath ./list.txt
```
- cf. text_data.json annotation format
```
{
  'YouTubeID_StartTime_EndTime': {
    'ko' : 'Parallel Korean Caption',
    'en' : 'Parallel English Caption',
}
```
<br>
<br>

# EXTRACTING VIDEO FEATURES 
: most VMT models does not have video feature extractor inside. we need to extract video features ourselves and use them as an input.
so, we need our own VIDEO FEATURE EXTRACTOR !

### *prerequisites*
```
pip install imageio
pip install --upgrade mxnet
pip install --upgrade gluoncv
```
<br>

### 0. Clone the Repo
you've already done it, right ? 

<br>

### 1. Extracting Video Features
- extract video features using the Inception-v1 I3D model pretrained on Kinetics 400 dataset and save them as .npy files. each video would be represented as a numpy array of size (1, num_of_segments, 1024).
```
python action_feature_extractor.py
```
<br>

### 2. Creating Action Labels
- some VMT models (i.e., [DEAR](https://www.sciencedirect.com/science/article/abs/pii/S0950705122002684)) takes video action labels as an input. we could create action labels also by using pretrained I3D model.
```
python action_label_extractor.py
```
### Then, our 'data' directory would be configured as following.
```
data
├── original_video 
│      ├── YouTubeID1.mp4
│      ├── YouTubeID2.mp4
│      └── .....  
│
├── original_subs
│      ├── YouTubeID1.ko.vtt
│      ├── YouTubeID1.en.vtt
│      ├── YouTubeID2.ko.vtt
│      ├── YouTubeID2.en.vtt
│      └── .....  
│
├── dataset
│      ├── video_data
│      │      ├── YouTubeID_StartTime_EndTime.mp4
│      │      ├── YouTubeID_StartTime_EndTime.mp4
│      │      └── .....
│      │
│      ├── action_features
│      │      ├── YouTubeID_StartTime_EndTime.npy
│      │      ├── YouTubeID_StartTime_EndTime.npy
│      │      └── .....
│      │
│      ├── action_labels.json
│      └── text_data.json
│
├── list.txt
├── utils.py
└── create_dataset.py
```
<br>
<br>
in progress...


