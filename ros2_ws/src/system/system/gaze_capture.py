# This is very similar to my original gaze capture code, using original GazeCapture model
# and inference with mediapipe library for all the detections

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import scipy.io as sio 
import numpy as np
from PIL import Image as pil_img
import os

from system.gaze.ITrackerModel import ITrackerModel
from custom_interfaces.msg import FaceMask

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ros_img
import cv2 
from cv_bridge import CvBridge 
from custom_interfaces.msg import Gaze

CHECKPOINTS_PATH = "/home/anja/ros2_ws/src/system/system/gaze"

LEFT_MAT = "/home/anja/ros2_ws/src/system/system/gaze/mean_left_224.mat"
FACE_MAT = "/home/anja/ros2_ws/src/system/system/gaze/mean_face_224.mat"
RIGHT_MAT = "/home/anja/ros2_ws/src/system/system/gaze/mean_right_224.mat"

faceMean = sio.loadmat(FACE_MAT, squeeze_me=True, struct_as_record=False)['image_mean'] 
eyeLeftMean = sio.loadmat(LEFT_MAT, squeeze_me=True, struct_as_record=False)['image_mean']
eyeRightMean = sio.loadmat(RIGHT_MAT, squeeze_me=True, struct_as_record=False)['image_mean']
one_size = 224

class SubtractMean(object):
    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        return tensor.sub(self.meanImg)

transformFace = transforms.Compose([
            transforms.Resize((one_size,one_size)),
            transforms.ToTensor(),
            SubtractMean(meanImg=faceMean)])

transformEyeL = transforms.Compose([
            transforms.Resize((one_size,one_size)),
            transforms.ToTensor(),
            SubtractMean(meanImg=eyeLeftMean)])

transformEyeR = transforms.Compose([
            transforms.Resize((one_size,one_size)),
            transforms.ToTensor(),
            SubtractMean(meanImg=eyeRightMean)])

model = ITrackerModel()
model = torch.nn.DataParallel(model)
model.cpu()


def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename, map_location=torch.device('cpu'))
    return state

saved = load_checkpoint()
if saved:
    print(
        'Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (
        saved['epoch'], saved['best_prec1']))
    state = saved['state_dict']
    try:
        model.module.load_state_dict(state)
    except:
        model.load_state_dict(state)
    epoch = saved['epoch']
    best_prec1 = saved['best_prec1']
else:
    print('Warning: Could not read checkpoint!')

class GazeCapture(Node):
    def __init__(self):
        super().__init__("gaze_capture")
        self.sub = self.create_subscription(ros_img, 'image_raw', self.listener_callback, 10)
        self.cv_bridge = CvBridge()
        self.sub_left_eye = self.create_subscription(ros_img, 'roi_left_eye', self.left_eye_callback, 10)
        self.sub_right_eye = self.create_subscription(ros_img, 'roi_right_eye', self.right_eye_callback, 10)
        self.sub_face = self.create_subscription(ros_img, 'roi_face', self.face_callback, 10)
        self.sub_face_mask = self.create_subscription(FaceMask, 'face_mask', self.mask_callback, 10)
        self.left_eye = None
        self.face = None
        self.face_mask = None
        self.right_eye = None
        self.predicted_pose_publisher = self.create_publisher(Gaze, 'predicted_pose', 10)

    def left_eye_callback(self, msg):
        self.left_eye = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def right_eye_callback(self, msg):
        self.right_eye = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def face_callback(self, data):
        self.face = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
    
    def mask_callback(self, data):
        grid_size = 25
        self.face_mask = np.array(data.data).reshape((grid_size, grid_size))
        

    def listener_callback(self, data):
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        if self.left_eye is not None and self.face is not None :
            self.gaze(self.left_eye, self.right_eye, self.face, self.face_mask, image)
            self.get_logger().info("Got all detections!")
        else:
            self.get_logger().info("Some detections are missing!")


    def gaze(self, roi_face, eye_left, eye_right, face_mask, frame):
        predicted_pose = np.array([0, 0])
        roi_face = transformFace(pil_img.fromarray(roi_face))
        eye_left = transformEyeL(pil_img.fromarray(eye_left))
        eye_right = transformEyeR(pil_img.fromarray(eye_right))
        face_mask = np.expand_dims(face_mask, 0)

        eye_left_t = torch.unsqueeze(eye_left, 0)
        eye_right_t = torch.unsqueeze(eye_right, 0)
        face_t = torch.unsqueeze(roi_face, 0)  
        face_mask_t = torch.FloatTensor(face_mask)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = model(face_t, eye_left_t, eye_right_t, face_mask_t)

        predicted_pose = output.detach().cpu().numpy()[0]

        print(predicted_pose)
        self.get_logger().info("Successfully predicted gaze!")
        msg = Gaze()
        msg.predicted_pose = predicted_pose.tolist() 
        self.predicted_pose_publisher.publish(msg)

def main():
    rclpy.init()
    node = GazeCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 
