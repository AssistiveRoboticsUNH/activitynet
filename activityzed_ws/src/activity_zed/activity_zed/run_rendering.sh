#!/bin/bash

# Set the ROS_DOMAIN_ID for the first process and run it in the background
# ROS_DOMAIN_ID=1 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashley/cam1/rosbag2_ashley/rosbag2_ashley &

# Set the ROS_DOMAIN_ID for the second process and run it in the background
# ROS_DOMAIN_ID=2 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashley/cam1/rosbag2_ashley_1/rosbag2_ashley_1 &



ROS_DOMAIN_ID=1 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam1/rosbag2_skylar_18_1/rosbag2_skylar_18_1 &
ROS_DOMAIN_ID=2 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam1/rosbag2_skylar_18/rosbag2_skylar_18 &
ROS_DOMAIN_ID=3 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam1/rosbag2_skylar_18_pill/rosbag2_skylar_18_pill &
ROS_DOMAIN_ID=4 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam1/rosbag2_ashleigh_18/rosbag2_ashleigh_18 &
ROS_DOMAIN_ID=5 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam1/rosbag2_ashleigh_18_pill/rosbag2_ashleigh_18_pill &
ROS_DOMAIN_ID=6 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam1/rosbag2_ashleigh_18_1/rosbag2_ashleigh_18_1 &


# Wait for both background processes to finish
wait

echo "All processes have completed."

