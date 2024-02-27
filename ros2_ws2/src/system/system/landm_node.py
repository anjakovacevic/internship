#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge 
import cv2
import mediapipe as mp
from custom_interfaces.msg import Landmarks, LandmarksArray


class LandmarksNode(Node):
    def __init__(self, name, pub):
        super().__init__(name)
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(LandmarksArray, pub, 10)
        self.cv_bridge = CvBridge()
        self.callback_function = None
        # self.keepData = None

    def get_publisher(self):
        return self.publisher 

    def set_callback(self, callback_function):
        self.callback_function = callback_function

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        
        if self.callback_function is not None:
            self.callback_function(image)
        else:
            # If callback_function is not provided, just display the frame
            cv2.imshow("Original_frame", image)
            cv2.waitKey(1)

