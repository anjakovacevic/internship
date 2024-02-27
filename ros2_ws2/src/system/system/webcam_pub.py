#!/usr/bin/env python3
'''
This Python script defines a ROS2 node that captures video frames
from a webcam or a video stream and publishes them to a ROS2 topic
named `image_raw`. It uses OpenCV to capture video frames and 
the `cv_bridge` package to convert the OpenCV image format to ROS2 
image message format (`sensor_msgs/msg/Image`). The `ImagePublisher` 
class initializes the video capture from the specified source (`url=0`
for the default webcam), sets the frame resolution, and contains a
timer callback that reads frames and publishes them at regular 
intervals. The main function initializes the ROS2 node, spins it to 
keep it running, and handles clean shutdown upon a keyboard interrupt.
'''
import rclpy                        
from rclpy.node import Node         
from sensor_msgs.msg import Image   
import cv2 
from cv_bridge import CvBridge, CvBridgeError

# url = 'http://10.10.104.56:8080/video'
url=0

class ImagePublisher(Node):
    def __init__(self, name):
        super().__init__(name)                                           
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)  
        self.timer = self.create_timer(0.1, self.timer_callback)         
        self.cap = cv2.VideoCapture(url) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cv_bridge = CvBridge()    
                               

    def timer_callback(self):
        ret, frame = self.cap.read()                                     
        
        if ret == True:                                                  
            self.publisher_.publish(
                self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8'))             

        self.get_logger().info('Publishing video frame')                 


def main(args=None):                                 
    rclpy.init(args=args)                            
    node = ImagePublisher("webcam_pub")        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()                                