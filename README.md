# VMT
### 1. CREATING DATASET
- create 'sample' directory for dataset 
```
mkdir sample
cd sample
```
- download vid/subs from youtube by using <a href="[https://github.com/ytdl-org/youtube-dl]" target="_blank">youtube-dl</a>

```
youtube-dl --get-id PLpqr-se75wgucnSktniG80CMTFuH0-3PY -i >> list.txt
youtube-dl -a list.txt -o '%(id)s.%(ext)s' --all-subs --rm-cache-dir
```
