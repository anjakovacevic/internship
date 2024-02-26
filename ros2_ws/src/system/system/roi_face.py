import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2
import mediapipe as mp
from rclpy.node import Node
import numpy as np
from custom_interfaces.msg import FaceMask

class FaceNode(Node):
    def __init__(self):
        super().__init__('face_det')
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.landmark_callback, 10)
        self.face_pub = self.create_publisher(Image, 'roi_face', 10)
        self.mask_pub = self.create_publisher(FaceMask, 'face_mask', 10) 
        self.cv_bridge = CvBridge()

    def landmark_callback(self, image):
        image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
            
        results = face_detection.process(img)

        if results.detections:
            face_bounding_box = self.get_face_boundaries(results.detections, image)
            x, y, w, h = face_bounding_box
            roi_face = image[y:y + h, x:x + w]

            if roi_face.shape[0] == 0:
                self.get_logger().info("No face detected.")
                return

            roi_face_msg = self.cv_bridge.cv2_to_imgmsg(roi_face, encoding='bgr8')
            self.face_pub.publish(roi_face_msg)
            self.get_logger().info("Published face ROI.")

            face_mask = self.getFaceGrid(image, (x, y, w, h))
            face_mask_msg = FaceMask()
            face_mask_data = face_mask.flatten().tolist()
            face_mask_msg.data = face_mask_data
            self.mask_pub.publish(face_mask_msg)
            self.get_logger().info("Published face mask.")
        else:
            self.get_logger().info("Waiting for face detection...")

    def get_face_boundaries(self, detections, image):
        for d in detections:
            bboxC = d.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
        return x,y,w,h

    def getFaceGrid(self, frame, face_bounding_box, grid_size=25):
        # Calculates the face grid from the face bounding box
        frameH, frameW, _ = frame.shape
        x, y, w, h = face_bounding_box
        scaleX = grid_size / frameW
        scaleY = grid_size / frameH

        grid = np.zeros((grid_size, grid_size))
        xLo = int(max(0, x * scaleX))
        yLo = int(max(0, y * scaleY))
        w = int(min(grid_size, w * scaleX))
        h = int(min(grid_size, h * scaleY))

        grid[yLo:yLo + h, xLo:xLo + w] = 1
        return grid

def main():
    rclpy.init()
    node = FaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 