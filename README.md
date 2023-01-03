![header](https://capsule-render.vercel.app/api?type=transparent&height=200&section=header&text=VMT%20for%20SUBS&fontSize=80&fontColor=020079)

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
    'en' : 'Parallel English Caption'},
    ...
}
```
<br>

# EXTRACTING VIDEO FEATURES 
: most VMT models do not have internal video feature extractor. we need to extract video features ourselves and use them as an input.
we need our own VIDEO FEATURE EXTRACTOR !

### *prerequisites*
```
pip install imageio
pip install --upgrade mxnet
pip install --upgrade gluoncv
```

### 0. Clone the Repo
- you've already done it, right ? 


### 1. Extracting Video Features
- extract video features using the Inception-v1 I3D model pretrained on Kinetics 400 dataset and save them as .npy files. each video would be represented as a numpy array of size (1, num_of_segments, 1024).
```
python action_feature_extractor.py
```

### 2. Creating Action Labels
- some VMT models (i.e., [DEAR](https://www.sciencedirect.com/science/article/abs/pii/S0950705122002684)) takes video action labels as an input. we could create action labels also by using pretrained I3D model.
```
python action_label_extractor.py
```
- cf. action_labels.json annotation format
```
{
  'YouTubeID_StartTime_EndTime': 
  [19, 17, 191, 171, 97],
   ...
}
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

# DEAR
## About DEAR
DEAR(Dual-lEvel bAck-tRanslation) is a model that investigated video-guided machine translation(VMT) task via dual-level back-translation. To be specific, it introduced sentence-level back-translation along with concept-level back-translation and implemented multi-pattern joint learning to improve translation performance.  
  
If you want to know more about DEAR, check out [the official website](https://kbs-2021.wixsite.com/dear) and [this post](https://twigfarm.notion.site/Video-guided-machine-translation-via-dual-level-back-translation-83f9d235c5d7405e92d98342e8a0fb6b).  
<br>

## Create Dataset
DEAR takes three elements as an input ; parallel sentence pairs, video action features, and video action labels.
Follow [this repo](https://github.com/twigfarm/letr-tool-KESVI) to construct appropriate dataset.  
<br>

## Train DEAR
### *prerequisites*
- basics  
: pytorch1.0.0+ recommended. I used torch==1.7.1+cu110
```
pip install torch 
```
- install konlpy + MeCab  
: this may be troublesome, but our friend google is always there for you. good luck !
```
# konlpy
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl  # install Java 1.8 or up

python3 -m pip install --upgrade pip
python3 -m pip install konlpy   
```
```
# MeCab
sudo apt-get install curl git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
### 0. Clone the Repo
```
! git clone https://github.com/twigfarm/letr-proto-DEAR.git
```
### 1. Train the Model
```
python train_circle.py
```
  
## Reference
- [Chen, S., Zeng, Y., Cao, D., & Lu, S. (2022). Video-guided machine translation via dual-level back-translation. Knowledge-Based Systems, 245, 108598.](https://www.sciencedirect.com/science/article/abs/pii/S0950705122002684)
- https://kbs-2021.wixsite.com/dear    

<br>
  
# VRET
## About VRET
VRET(Visual Relationship-Enhanced Transformer) is a model that investigated video-guided machine translation(VMT) task via visual relationship-enhanced transformer by constructing a semantic–visual relational graph as a cross-modal bridge. To be specific, graph convolutional network was deployed to capture the relationship among visual semantics to improve translation performance.  
  
If you want to know more about VRET, check out [the official website](https://eswa-2021.wixsite.com/vret) and [this post](https://twigfarm.notion.site/Vision-talks-Visual-relationship-enhanced-transformer-for-video-guided-machine-translation-7bc8e6947ee04dcd8143635b682570a0).  
<br>

## Create Dataset
DEAR takes three elements as an input ; parallel sentence pairs, scene nodes, and scene graphs.
Follow [this repo](https://github.com/twigfarm/letr-tool-KESVI) to construct parallel sentence pairs, and follow [this repo](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to create scene nodes and scene graphs.   
<br>

## Train VRET
### *prerequisites*
- basics  
: pytorch1.0.0+ recommended. I used torch==1.7.1+cu110
```
pip install torch 
```
- install TrTokenizer  
```
pip install trtokenizer
```
### 0. Clone the Repo
```
! git clone https://github.com/twigfarm/letr-proto-VRET.git
```
### 1. Train the Model
```
python train.py
```
  
## Reference
- [Chen, S., Zeng, Y., Cao, D., & Lu, S. (2022). Vision talks: Visual relationship-enhanced transformer for video-guided machine translation. Expert Systems with Applications, 209, 118264.](https://www.sciencedirect.com/science/article/abs/pii/S0957417422014051)
- https://eswa-2021.wixsite.com/vret  



