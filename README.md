# VMT
### 1. CREATING DATASET
- create 'sample' directory for dataset 
```
mkdir sample
cd sample
```
- downloading vid/subs from youtube by using youtube-dl
```
youtube-dl --get-id PLpqr-se75wgucnSktniG80CMTFuH0-3PY -i >> list.txt
youtube-dl -a list.txt -o '%(id)s.%(ext)s' --all-subs --rm-cache-dir
```
