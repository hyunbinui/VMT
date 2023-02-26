- data
    ├── video_features
    │     ├── scene_node
    │     │       ├── YouTubeID_StartTime_EndTime.npy
    │     │       ├── YouTubeID_StartTime_EndTime.npy
    │     │       └── ...    
    │     └── scene_v_graph
    │             ├── YouTubeID_StartTime_EndTime.npy
    │             ├── YouTubeID_StartTime_EndTime.npy
    │             └── ...    
    ├── train_data.json
    ├── test_data.json
    └── valid_data.json


[Annotation Style]

- {train, test, valid}_data.json
{
  'YouTubeID_StartTime_EndTime': {
    'ko' : 'Parallel Korean Caption',
    'en' : 'Parallel English Caption'},
    ...
}
