# To check if eye detection works, add this to setup.py:
# 'eyes_viewer = system.eye_viewer:main',
# And run it after image_raw and get_eyes, or after using the launch_file

import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image

class EyesViewer(Node):
    def __init__(self, name):
        super().__init__(name)
        self.cv_bridge = CvBridge()

        # Create subscribers for left and right eye ROIs
        self.sub_left_eye = self.create_subscription(Image, 'roi_left_eye', self.left_eye_callback, 10)
        self.sub_right_eye = self.create_subscription(Image, 'roi_right_eye', self.right_eye_callback, 10)

    def left_eye_callback(self, msg):
        # Process the left eye ROI image
        # Example: Display the image using OpenCV
        left_eye_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if left_eye_image.size > 0:
            cv2.imshow('Left Eye ROI', left_eye_image)  # Corrected window name
        else:
            self.get_logger().info("Empty left eye image!")
        cv2.waitKey(1)  # Adjust the delay as needed

    def right_eye_callback(self, msg):
        # Process the right eye ROI image
        # Example: Display the image using OpenCV
        right_eye_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if right_eye_image.size > 0:
            cv2.imshow('Right Eye ROI', right_eye_image)
        else:
            self.get_logger().info("Empty right eye image!")
        cv2.waitKey(1)  # Adjust the delay as needed


def main():
    rclpy.init()
    node = EyesViewer('eyes_viewer')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
