### Play rosbag file

```bash
cd the_directoy_that_contains_the_rosbag_file
ros2 bag play rosbag2_akash_0.db3
```

### Save image topic as video file.

```bash
cd ~/activitynet/activityzed_ws/src/activity_zed/activity_zed/
python zed_sub.py    # a video window will pop up. Close this window when rosbag play stop in the previous terminal.
```

Saved videos are stored "~/activityzed_ws/videos/mm_dd_yyyy_hh_mm"

Inside "~/activityzed_ws/videos/mm_dd_yyyy_hh_mm" create a text file called "segments.txt"

Now, watche the video and store labelled activity segments in the "segments.txt" file.
Sample contents in the "segments.txt" file:
``` 
0-200,eating
230-430,not_eating
500-700,eating
800-1000,not_eating
```
Keep 200 frame for each activity.


### Run Inference

```bash
cd /ros2_ws/src/activity_zed/activity_zed
python zed_inf.py
```


### View Inference Results

```bash
ros2 topic echo /activity
```

