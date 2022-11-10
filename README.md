# VMT for SUBS 
## CREATING DATASET 
#### 1. Downloading Youtube Playlist
- create 'dataset' directory for dataset 
```
mkdir dataset
cd dataset
```
- download videos / subtitles from youtube by using [youtube-dl](https://github.com/ytdl-org/youtube-dl)

```
youtube-dl --get-id [playlist link] -i >> list.txt
youtube-dl -a list.txt -o '%(id)s.%(ext)s' --rm-cache-dir --all-subs 
```
<align = center>↓ 
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
├── video 1
├── video 2
├── ....
│
├── videoid.ko.vtt
├── videoid.en.vtt
├── ...
│
├── list.txt
└── sample.json
```
<br>
<br>

in progress...


