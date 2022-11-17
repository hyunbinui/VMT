![header](https://capsule-render.vercel.app/api?type=transparent&height=200&section=header&text=VMT%20for%20SUBS&fontSize=80&fontColor=0000ff)

# CREATING DATASET 
### *prerequisites*
```
pip install moviepy
pip install imageio_ffmpeg
pip install webvtt-py
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
- then, the 'data' directory would be configured as following :
```
data
├── original_video 
│      ├── videoid1.mp4
│      ├── videoid2.mp4
│      └── .....  
│
├── original_subs
│      ├── videoid1.ko.vtt
│      ├── videoid1.en.vtt
│      ├── videoid2.ko.vtt
│      ├── videoid2.en.vtt
│      └── .....  
│
├── dataset
│      ├── video_data
│      │      ├── videoid_starttime_endtime.mp4
│      │      ├── videoid_starttime_endtime.mp4
│      │      └── .....
│      │
│      └── text_data.json
│
├── list.txt
├── utils.py
└── create_dataset.py
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

in progress...


