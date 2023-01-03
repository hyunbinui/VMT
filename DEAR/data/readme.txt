You must make data repos like following.

- data
    ├── label
    │     └── action_label.json
    ├── action_feature
    │     ├── YouTubeID_StartTime_EndTime.npy
    │     ├── YouTubeID_StartTime_EndTime.npy
    │     └── ...
    ├── train_data.json
    ├── test_data.json
    └── valid_data.json


[Annotation Style]
- action_label.json
{
  'YouTubeID_StartTime_EndTime': 
  [19, 17, 191, 171, 97],
   ...
}

- {train, test, valid}_data.json
{
  'YouTubeID_StartTime_EndTime': {
    'ko' : 'Parallel Korean Caption',
    'en' : 'Parallel English Caption'},
    ...
}
