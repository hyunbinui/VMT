You must make data repos like following.

- data
    ├── label
    │     └── sample_data.csv
    └── video_features
          ├── scene_node
          │       ├── YouTubeID_StartTime_EndTime.npy
          │       ├── YouTubeID_StartTime_EndTime.npy
          │       └── ...    
          └── scene_v_graph
                  ├── YouTubeID_StartTime_EndTime.npy
                  ├── YouTubeID_StartTime_EndTime.npy
                  └── ...    

[Annotation Style]

- {train, test, valid}_data.csv
vid_id,tr,en
eyhzdC936uk_15_27@1,köpekle oynayan bir çocuk,a boy playing with dog