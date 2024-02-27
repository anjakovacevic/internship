'''
This code creates a ROS2 node that subscribes to a video feed, 
detects face landmarks using MediaPipe, and publishes these landmarks
to another ROS2 topic. It employs cv_bridge to convert ROS2 image
messages into OpenCV formats for processing. Custom callback 
functionality allows for flexible image processing within the node. 
Detected landmarks are encapsulated in a custom ROS2 message and 
published for further use. The application supports real-time 
face landmark detection and publishing in a ROS2 ecosystem.
'''

import rclpy
from system.landm_node import LandmarksNode

from cv_bridge import CvBridge 
import cv2
import mediapipe as mp
from std_msgs.msg import String
from custom_interfaces.msg import Landmarks, LandmarksArray

def landmark_callback(image, node):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        
    results = face_detection.process(img)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    results2 = face_mesh.process(img)

    if results2.multi_face_landmarks:
        for face_landmarks in results2.multi_face_landmarks:
            landmark_array_msg = LandmarksArray()
            for landmark in face_landmarks.landmark:
                landmark_msg = Landmarks()
                landmark_msg.x = landmark.x
                landmark_msg.y = landmark.y
                landmark_msg.z = landmark.z
                landmark_array_msg.landmarks.append(landmark_msg)
            node.get_logger().info(f"Publishing {len(landmark_array_msg.landmarks)} landmarks")
            node.publisher.publish(landmark_array_msg)


def main():
    rclpy.init()
    node = LandmarksNode(name='all_landmarks', pub='landm_topic')
    node.set_callback(lambda img: landmark_callback(img, node))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 