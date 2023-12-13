import os
import numpy as np
import scipy.io as sio

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from imutils import face_utils
import imutils
import mediapipe as mp
import cv2
import argparse
import pygame
import sys

from ITrackerModel import ITrackerModel

CHECKPOINTS_PATH = 'C:/Users/Anja/kod/fixing-gazecapture-master'
# CHECKPOINTS_PATH = 'C:/Users/Anja/kod/GazeCapture-master/pytorch'
# CHECKPOINTS_PATH = 'C:/Users/Anja/kod/gaze_drugi_laptop/zip'

# from canonical_face_model_uv_visualization - Mediapipe incides for face mesh
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 243, 112, 26, 22, 23, 24, 110, 25]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 463, 341, 256, 252, 253, 254, 339, 255]

faceMean = sio.loadmat('mean_face_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']
eyeLeftMean = sio.loadmat('mean_left_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']
eyeRightMean = sio.loadmat('mean_right_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']
one_size = 224

pygame.init()

window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Gaze Tracking Dot")

# Set up the dot
dot_radius = 10
dot_color = (255, 0, 0)  # Red color, you can adjust this
dot_position = [0, 0]  # Initial position

grey = (192, 192, 192)

clock = pygame.time.Clock()

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


def get_face_boundaries(detections, image):
    for d in detections:
        bboxC = d.location_data.relative_bounding_box
        ih, iw, ic = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
    return x,y,w,h

def get_roi_eyes(landmarks, image):
    left_eye_landmarks = [landmarks.landmark[i] for i in (130, 247, 161, 160, 159, 158, 157, 173, 243, 112, 26, 22, 23, 24, 110, 25)]
    right_eye_landmarks = [landmarks.landmark[i] for i in (359, 467, 388, 387, 386, 385, 384, 398, 463, 341, 256, 252, 253, 254, 339, 255)]

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
    
    # cv2.rectangle(image, (left_min_x, left_min_y), (left_max_x, left_max_y), (0, 255, 0), 2)
    # cv2.rectangle(image, (right_min_x, right_min_y), (right_max_x, right_max_y), (0, 255, 0), 2)
    roi_l_eye = image[left_min_y:left_max_y, left_min_x:left_max_x]
    roi_r_eye = image[right_min_y:right_max_y, right_min_x:right_max_x]
    # for landmark in left_eye_coords + right_eye_coords:
    #     cv2.circle(image, landmark, 5, (0, 255, 0), -1)  # Draw a filled circle

    return roi_l_eye, roi_r_eye


def getFaceGrid(frame, face_bounding_box, grid_size=25):
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


def get_all_detections(image):
    # This is just for face
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img)
    if results.detections:
        face_bounding_box = get_face_boundaries(results.detections, image)
        x, y, w, h = face_bounding_box
        roi_face = image[y:y + h, x:x + w]
        # cv2.imwrite("Face.jpg", roi_face)
        
        # Face grid
        face_mask = getFaceGrid(image, (x, y, w, h))
    
    # Now the eyes
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    results = face_mesh.process(img)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        roi_l_eye, roi_r_eye = get_roi_eyes(landmarks, image)
        # cv2.imwrite("l_eye.jpg", roi_l_eye)
        # cv2.imwrite("r_eye.jpg", roi_r_eye)
    face_mesh.close()
    
    return roi_face, roi_l_eye, roi_r_eye, face_mask



def headpose_est(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5
    )

    # 3D face model
    face_3d = np.array([
        [0.0, 0.0, 0.0],            # Nose tip
        [0.0, -330.0, -65.0],       # Chin
        [-225.0, 170.0, -135.0],    # Left eye left corner
        [225.0, 170.0, -135.0],     # Right eye right corner
        [-150.0, -150.0, -125.0],   # Left Mouth corner
        [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

    # We need eye coords for drawing the headpose estimation
    # Reposition left eye corner to be the origin
    leye_3d = np.array(face_3d)
    leye_3d[:,0] += 225
    leye_3d[:,1] -= 175
    leye_3d[:,2] += 135

    # Reposition right eye corner to be the origin
    reye_3d = np.array(face_3d)
    reye_3d[:,0] -= 225
    reye_3d[:,1] -= 175
    reye_3d[:,2] += 135
        
    img_h, img_w, img_c = frame.shape
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ])

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    results = face_mesh.process(frame)

    for face_landmarks in results.multi_face_landmarks:
        face_2d = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks.landmark]

        # Get relevant landmarks for head pose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        # Solve PnP for left and right eyes
        # slovePnP solves the Perspective-n-Point (PnP) problem. It estimates the pose of an object given its 3D coordinates 
        # in the world (objectPoints) and their 2D coordinates in the image (imagePoints). 
        # The output includes the rotation and translation vectors.
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get rotational matrix from rotational vector
        # Rodrigues function in OpenCV converts a rotation vector (rvec) into a rotation matrix (rmat). 
        # It's commonly used to switch between the compact representation of 3D rotations.
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        # Draw axis of rotation for left eye
        # This function projects 3D points (objectPoints) into 2D image points given the camera
        # parameters (cameraMatrix, distCoeffs) and the pose of the object in the scene (rvec, tvec).
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)

        # Draw the axis on the image
        # ravel is used to flatten multi-dimensional arrays into a 1D array. 
        cv2.line(frame, tuple(np.ravel(l_axis[0]).astype(np.int32)), tuple(np.ravel(l_axis[1]).astype(np.int32)), (0, 200, 0), 3)
        cv2.line(frame, tuple(np.ravel(l_axis[1]).astype(np.int32)), tuple(np.ravel(l_axis[2]).astype(np.int32)), (0, 0, 200), 3)

        cv2.line(frame, tuple(np.ravel(r_axis[0]).astype(np.int32)), tuple(np.ravel(r_axis[1]).astype(np.int32)), (0, 200, 0), 3)
        cv2.line(frame, tuple(np.ravel(r_axis[1]).astype(np.int32)), tuple(np.ravel(r_axis[2]).astype(np.int32)), (0, 0, 200), 3)



def test_webcam(model, id):
    video_capture = cv2.VideoCapture(id)

    while (True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        ret, frame = video_capture.read()
        if not ret:
            print("Camera doesn't work.")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        frame = imutils.resize(frame, width=500)
        data_from_frame = get_all_detections(frame)
        
        eye_left, eye_right, roi_face, face_mask = data_from_frame
        # print(f"Type of eye_left: {type(eye_left)}, Type of eye_right: {type(eye_right)}, Type of roi_face: {type(roi_face)}")

        roi_face = transformFace(Image.fromarray(roi_face))
        eye_left = transformEyeL(Image.fromarray(eye_left))
        eye_right = transformEyeR(Image.fromarray(eye_right))
        face_mask = np.expand_dims(face_mask, 0)

        # print(f"Type of eye_left: {type(eye_left)}, Type of eye_right: {type(eye_right)}, Type of roi_face: {type(roi_face)}")
        
        eye_left_t = torch.unsqueeze(eye_left, 0)
        eye_right_t = torch.unsqueeze(eye_right, 0)
        face_t = torch.unsqueeze(roi_face, 0)  
        face_mask_t = torch.FloatTensor(face_mask)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = model(face_t, eye_left_t, eye_right_t, face_mask_t)
        predicted_pose = output.detach().cpu().numpy()[0]

        headpose_est(frame)

        # left_x, left_y, left_w, left_h = left_eye_bounding_box
        # right_x, right_y, right_w, right_h = right_eye_bounding_box
        # frame = cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 2)
        # frame = cv2.rectangle(frame, (right_x, right_y), (right_x + right_w, right_y + right_h), (0, 255, 0), 2)

        # frame = cv2.resize(frame, (one_size, one_size))
        # frame = cv2.circle(frame, (center[0] + int(predicted_pose[0][0]), center[1] + int(predicted_pose[0][1])), 5, (0, 255, 0), -1)
        print(predicted_pose)
        
        
        x_gaze_min, x_gaze_max = -10, 14
        y_gaze_min, y_gaze_max = -14, 14

        pygame_x = int((-1*predicted_pose[0] - x_gaze_min) / (x_gaze_max - x_gaze_min) * window_size[0])
        pygame_y = int((-1*predicted_pose[1] - y_gaze_min) / (y_gaze_max - y_gaze_min) * window_size[1])

        # dot_position = [int((-1* predicted_pose[0] + 13) / (13 + 13) * window_size[0]), int((-1*predicted_pose[1] + 13) / (13 + 13) * window_size[1])]
        dot_position = [pygame_x, pygame_y]
        print(predicted_pose[0], predicted_pose[1], dot_position)
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, dot_color, dot_position, dot_radius)

        # Draw thin grey lines for x and y axes
        pygame.draw.line(screen, grey, (0, window_size[1] // 2), (window_size[0], window_size[1] // 2), 1)
        pygame.draw.line(screen, grey, (window_size[0] // 2, 0), (window_size[0] // 2, window_size[1]), 1)

        pygame.display.flip()
        clock.tick(60)       

        cv2.imshow("Gaze point", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

model = ITrackerModel()
model = torch.nn.DataParallel(model)
model.cuda()

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

# my phone link -> 'http://10.10.104.56:8080/video'
parser = argparse.ArgumentParser()
parser.add_argument('id', type=str, help='Are you using wecam or phone')
args = parser.parse_args()

if args.id == '1' or args.id == '0' or args.id == '2':
    id = int(args.id)
    print(id, type(id))
else:
    id = args.id
    print(id)

test_webcam(model, id)