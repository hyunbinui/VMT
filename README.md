# VMT for SUBS
### [ CREATING DATASET ]
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
<br>

#### 2. Constructing Dataset
- construct the text pair dataset and video dataset by running the 'create_dataset.ipynb' file

<br>


in progress...


