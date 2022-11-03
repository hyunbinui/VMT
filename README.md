# VMT for SUBS
### 1. CREATING DATASET
- create 'sample' directory for dataset 
```
mkdir sample
cd sample
```
- download videos / subtitles from youtube by using [youtube-dl](https://github.com/ytdl-org/youtube-dl)

```
youtube-dl --get-id [playlist link] -i >> list.txt
youtube-dl -a list.txt -o '%(id)s.%(ext)s' --all-subs --rm-cache-dir
```

<br>



😧 in progress,,


