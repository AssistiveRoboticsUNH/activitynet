import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import rclpy
from rclpy.node import Node
 
from threading import Thread
import time
import numpy as np 
from std_msgs.msg import Bool
 
import cv2
from matplotlib import pyplot as plt
import pickle
import datetime
import os 
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ViewImageTopic(Node):

    def __init__(self):
        super().__init__('ViewImageTopic')
        
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Replace with your actual image topic
            self.image_callback,
            10  # QoS profile
        )
        self.subscription

        self.cv_bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow('ROS2 Image Viewer', cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ViewImageTopic() 
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

