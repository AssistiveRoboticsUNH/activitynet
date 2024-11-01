#!/bin/bash

ROS_DOMAIN_ID=1 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam4/rosbag2_skylar_18_1/rosbag2_skylar_18_1 &
ROS_DOMAIN_ID=2 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam4/rosbag2_skylar_18/rosbag2_skylar_18 &
ROS_DOMAIN_ID=3 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/skylar/cam4/rosbag2_skylar_18_pill/rosbag2_skylar_18_pill &
ROS_DOMAIN_ID=4 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam4/rosbag2_ashleigh_18/rosbag2_ashleigh_18 &
ROS_DOMAIN_ID=5 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam4/rosbag2_ashleigh_18_pill/rosbag2_ashleigh_18_pill &
ROS_DOMAIN_ID=6 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashleigh/cam4/rosbag2_ashleigh_18_1/rosbag2_ashleigh_18_1 &
ROS_DOMAIN_ID=7 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashley/cam4/rosbag2_ashley/rosbag2_ashley &
ROS_DOMAIN_ID=8 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/ashley/cam4/rosbag2_ashley_1/rosbag2_ashley_1 &
ROS_DOMAIN_ID=9 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/lauren/cam4/rosbag2_lauren/rosbag2_lauren &
ROS_DOMAIN_ID=10 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/lauren/cam4/rosbag2_lauren_1/rosbag2_lauren_1 &
ROS_DOMAIN_ID=11 python3 zed_sub.py --rosbag_path /media/ns/Seagate/oct18/lauren/cam4/rosbag2_lauren_2/rosbag2_lauren_2 &


# Wait for both background processes to finish
wait

echo "All processes have completed."

