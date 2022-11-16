![header](https://capsule-render.vercel.app/api?type=transparent&height=200&section=header&text=VMT%20for%20SUBS&fontSize=80&fontColor=0000ff)

## CREATING DATASET 
#### *prerequisites*
```
pip install moviepy
pip install imageio_ffmpeg
pip install webvtt-py
```
#### 1. Downloading Youtube Videos
- create 'original_video' directory inside 'data' directory for Youtube videos 
```
mkdir original_video
cd original_video
```
- download videos / subtitles from youtube by using [youtube-dl](https://github.com/ytdl-org/youtube-dl)

```
youtube-dl --get-id [playlist link] -i >> list.txt
youtube-dl -a list.txt --write-srt --sub-lang en,ko -o '%(id)s.%(ext)s' --skip-download -i 
```
- if youtube-dl is way too slow, try using [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading videos

```
yt-dlp -a list.txt -o '%(id)s.%(ext)s' -S ext:mp4:m4a -i
# youtube-dl -a list.txt --write-srt --sub-lang en,ko -o '%(id)s.%(ext)s' --skip-download -i 
```
<br>

#### 2. Constructing Dataset
- download 'create_dataset.ipynb' file to the 'dataset' directory
- construct the text pair and video dataset by running the 'create_dataset.ipynb' file <br>
  then, the 'dataset' directory would be configured as following :
```
dataset
├── create_dataset.ipynb
├── videos 
│      ├── videoid_starttime_endtime.mp4
│      ├── videoid_starttime_endtime.mp4
│      └── .....  
│
├── videoid.mp4
├── videoid.mp4
├── ....
│
├── videoid.ko.vtt
├── videoid.en.vtt
├── ...
│
├── list.txt
└── text_data.json
```
<br>
<br>

in progress...


