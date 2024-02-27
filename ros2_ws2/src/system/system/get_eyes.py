import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge 
import cv2
import numpy as np
from custom_interfaces.msg import Landmarks, LandmarksArray

class EyesSub(Node):
    def __init__(self, name):
        super().__init__(name)
        self.sub_image = self.create_subscription(Image, 'image_raw', self.get_and_publish_roi_eyes, 10)
        self.sub_landmarks = self.create_subscription(LandmarksArray, 'landm_topic', self.landmarks_callback, 10)
        self.pub_left_eye = self.create_publisher(Image, 'roi_left_eye', 10)
        self.pub_right_eye = self.create_publisher(Image, 'roi_right_eye', 10)

        self.cv_bridge = CvBridge()
        self.eye_msg = String()

        self.landmarks = LandmarksArray()

        # To check if landmarks were recieved
        self.timer = self.create_timer(5.0, self.check_landmarks)
        self.landmarks_received = False

    def landmarks_callback(self, msg):
        self.landmarks = msg
        self.landmarks_received = True

    def check_landmarks(self):
        if not self.landmarks_received:
            self.get_logger().info("Still waiting for landmarks...")


    def get_and_publish_roi_eyes(self, msg):
        if not self.landmarks_received:
            self.get_logger().info("Didn't get any landmarks")
            return

        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   

        roi_l_eye, roi_r_eye = self.get_roi_eyes(image)
        if roi_l_eye.shape[0] == 0 or roi_r_eye.shape[0] == 0:
            self.get_logger().info("No eyes detected.")
            return

        roi_l_eye_msg = self.cv_bridge.cv2_to_imgmsg(roi_l_eye, encoding='bgr8')
        self.pub_left_eye.publish(roi_l_eye_msg)

        roi_r_eye_msg = self.cv_bridge.cv2_to_imgmsg(roi_r_eye, encoding='bgr8')
        self.pub_right_eye.publish(roi_r_eye_msg)


        self.get_logger().info("Published left and right eye ROIs.")


    def get_roi_eyes(self, image):
        # Define the indices for the left and right eyes
        left_eye_landmarks = [self.landmarks.landmarks[i] for i in (130, 247, 161, 160, 159, 158, 157, 173, 243, 112, 26, 22, 23, 24, 110, 25)]
        right_eye_landmarks = [self.landmarks.landmarks[i] for i in(359, 467, 388, 387, 386, 385, 384, 398, 463, 341, 256, 252, 253, 254, 339, 255)]

        # Convert normalized coordinates to image coordinates
        height, width, _ = image.shape
        left_eye_coords = [(int(point.x * width), int(point.y * height)) for point in left_eye_landmarks]
        right_eye_coords = [(int(point.x * width), int(point.y * height)) for point in right_eye_landmarks]
        left_min_x = min(left_eye_coords, key=lambda x: x[0])[0]
        left_min_y = min(left_eye_coords, key=lambda x: x[1])[1]
        left_max_x = max(left_eye_coords, key=lambda x: x[0])[0]
        left_max_y = max(left_eye_coords, key=lambda x: x[1])[1]
        left_size = max(left_max_x - left_min_x, left_max_y - left_min_y)
        left_min_x = max(0, left_min_x - (left_size - (left_max_x - left_min_x)) // 2)
        left_min_y = max(0, left_min_y - (left_size - (left_max_y - left_min_y)) // 2)
        left_max_x = min(width, left_max_x + (left_size - (left_max_x - left_min_x)) // 2)
        left_max_y = min(height, left_max_y + (left_size - (left_max_y - left_min_y)) // 2)

        # Calculate bounding box coordinates for right eye
        right_min_x = min(right_eye_coords, key=lambda x: x[0])[0]
        right_min_y = min(right_eye_coords, key=lambda x: x[1])[1]
        right_max_x = max(right_eye_coords, key=lambda x: x[0])[0]
        right_max_y = max(right_eye_coords, key=lambda x: x[1])[1]
        right_size = max(right_max_x - right_min_x, right_max_y - right_min_y)
        right_min_x = max(0, right_min_x - (right_size - (right_max_x - right_min_x)) // 2)
        right_min_y = max(0, right_min_y - (right_size - (right_max_y - right_min_y)) // 2)
        right_max_x = min(width, right_max_x + (right_size - (right_max_x - right_min_x)) // 2)
        right_max_y = min(height, right_max_y + (right_size - (right_max_y - right_min_y)) // 2)
        
        cv2.rectangle(image, (left_min_x, left_min_y), (left_max_x, left_max_y), (0, 255, 0), 2)
        cv2.rectangle(image, (right_min_x, right_min_y), (right_max_x, right_max_y), (0, 255, 0), 2)
        roi_l_eye = image[left_min_y:left_max_y, left_min_x:left_max_x]
        roi_r_eye = image[right_min_y:right_max_y, right_min_x:right_max_x]
        # for landmark in left_eye_coords + right_eye_coords:
        #     cv2.circle(image, landmark, 5, (0, 255, 0), -1)  # Draw a filled circle
        # cv2.imshow("eye", roi_l_eye)

        return roi_l_eye, roi_r_eye

    def parse_landmarks_data(self, landmarks_data):
        landmarks_list = []

        # Split the landmarks data into individual entries
        entries = landmarks_data.split(';')
        
        for entry in entries:
            if entry.strip():  # Check if the entry is not empty
                # Split each entry into index, x, and y
                parts = entry.split(',')
                index = int(parts[0].split(':')[1].strip())
                x = int(parts[1].split(':')[1].strip())
                y = int(parts[2].split(':')[1].strip())
                
                landmarks_list.append({"index": index, "x": x, "y": y})

        return landmarks_list

def main():
    rclpy.init()
    node = EyesSub('get_eyes')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
