# VMT for SUBS
### 1. CREATING DATASET
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
<br>


😧 in progress,,


