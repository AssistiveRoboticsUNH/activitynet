#!/bin/bash

ROS_DOMAIN_ID=1 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam3/rosbag2_katy_nov_pill &
ROS_DOMAIN_ID=2 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam3/rosbag2_katy_nov_2 &
ROS_DOMAIN_ID=3 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam3/rosbag2_katy_nov_1 &
ROS_DOMAIN_ID=4 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam4/rosbag2_katy_nov_pill &
ROS_DOMAIN_ID=5 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam4/rosbag2_katy_nov_2 &
ROS_DOMAIN_ID=6 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam4/rosbag2_katy_nov_1 &
ROS_DOMAIN_ID=7 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam2/rosbag2_katy_nov_pill &
ROS_DOMAIN_ID=8 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam2/rosbag2_katy_nov_2 &
ROS_DOMAIN_ID=9 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam2/rosbag2_katy_nov_1 &
ROS_DOMAIN_ID=10 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam1/rosbag2_katy_nov_pill &
ROS_DOMAIN_ID=11 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam1/rosbag2_katy_nov_2 &
ROS_DOMAIN_ID=12 python3 /home/ns/activitynet/activityzed_ws/src/activity_zed/activity_zed/zed_sub.py --rosbag_path /media/ns/Seagate/nov6/katy/cam1/rosbag2_katy_nov_1 &


# Wait for all the background processes to finish
wait

echo "All processes have completed."

