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
import scipy.ndimage

from ITrackerModel import ITrackerModel

# CHECKPOINTS_PATH = 'C:/Users/Anja/kod/fixing-gazecapture-master'
CHECKPOINTS_PATH = 'C:/Users/Anja/kod/GazeCapture-master/pytorch'
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
dot_color = (255, 0, 0) 
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

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    results2 = face_mesh.process(img)

    if results.detections and results2.multi_face_landmarks:
        face_bounding_box = get_face_boundaries(results.detections, image)
        x, y, w, h = face_bounding_box
        roi_face = image[y:y + h, x:x + w]
        # Eyes
        landmarks = results2.multi_face_landmarks[0]
        roi_l_eye, roi_r_eye = get_roi_eyes(landmarks, image)
        # Face grid
        face_mask = getFaceGrid(image, (x, y, w, h))

        return roi_face, roi_l_eye, roi_r_eye, face_mask
    else: 
        return None, None, None, None


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

def kalman_init(dt, init=None):
    if init is None :
        x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
    else:
        x = init
    
    P = np.matrix([[0.01, 0.0, 0.0, 0.0],
                   [0.0, 0.01, 0.0, 0.0],
                   [0.0, 0.0, 7.0, 0.0],
                   [0.0, 0.0, 0.0, 7.0]])
    
    A = np.matrix([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    
    sj = 0.2
    Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
               [0, (dt**4)/4, 0, (dt**3)/2],
               [(dt**3)/2, 0, dt**2, 0],
               [0, (dt**3)/2, 0, dt**2]]) * sj**2
    I = np.eye(4)

    return x, P, A, Q, I 
    

def distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def gaus(x, y, num, sigma):
    if len(x) > num:
        recent_x = x[-num:]
        recent_y = y[-num:]
    else:
        recent_x = x
        recent_y = y
    
    x = scipy.ndimage.gaussian_filter(recent_x, sigma)
    y = scipy.ndimage.gaussian_filter(recent_y, sigma)

    return x[-1], y[-1]


def test_webcam(model, id):
    video_capture = cv2.VideoCapture(id)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    state, P, A, Q, I = kalman_init(1/fps)
    xt = []
    yt = []
    dxt= []
    dyt= []
    x_measured = []
    y_measured = []
    threshold = 3
    predicted_pose = np.array([0, 0])
    q=0
    x_gaus = []
    y_gaus = []

    while (True):
        q+=1
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

        if data_from_frame[0] is not None:
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

            previous = predicted_pose
            predicted_pose = output.detach().cpu().numpy()[0]


            # left_x, left_y, left_w, left_h = left_eye_bounding_box
            # right_x, right_y, right_w, right_h = right_eye_bounding_box
            # frame = cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 2)
            # frame = cv2.rectangle(frame, (right_x, right_y), (right_x + right_w, right_y + right_h), (0, 255, 0), 2)

            # frame = cv2.resize(frame, (one_size, one_size))
            # frame = cv2.circle(frame, (center[0] + int(predicted_pose[0][0]), center[1] + int(predicted_pose[0][1])), 5, (0, 255, 0), -1)
            print(predicted_pose)

            # This part is Kalman filtering of the gaze
            predicted_value = predicted_pose.reshape(-1, 1)

            x_measured.append(float(predicted_pose[0]))
            y_measured.append(float(predicted_pose[1]))

            # Apply Gaussian filter to the measurements
            xg, yg = gaus(x_measured, y_measured, 5, 2)

            if q % 10 == 0 :
                state, P, A, Q, I = kalman_init(1/fps, state)
                
            state = A * state
            P = A * P * A.T + Q 
            # Measurement Update (Correction)q
            H = np.matrix([[1.6, 0.0, 0.0, 0.0],
                        [0.0, 1.6, 0.0, 0.0]])
            # Measurement covariance matrix R - assuming some measurement noise
            measurement_noise = 1.0
            R = np.matrix([[measurement_noise, 0.0],
                        [0.0, measurement_noise]])
            # Compute the Kalman Gain
            S = H * P * H.T + R
            K = (P * H.T) * np.linalg.pinv(S)
            # Update the estimate via z
            Z = predicted_value.reshape(H.shape[0], 1)
            y = Z - (H * state)
            state = state + (K * y)
            # Update the error covariance
            P = (I - (K * H)) * P  

            # Dynamic selection of the filter or the measurement
            change = distance(predicted_pose[0], previous[0], predicted_pose[1], previous[1])
            if change > threshold:
                x = state[0]
                y = state[1]
            else:
                # x = predicted_pose[0]
                # y = predicted_pose[1]    
                x = xg
                y = yg
                x_gaus.append(xg)
                y_gaus.append(yg)

            xt.append(float(state[0]))
            yt.append(float(state[1]))
            dxt.append(float(state[2]))
            dyt.append(float(state[3]))

            # This part is for displaying the result of the gaze
            x_gaze_min, x_gaze_max = -10, 14
            y_gaze_min, y_gaze_max = -14, 14

            pygame_x = int((-1*x - x_gaze_min) / (x_gaze_max - x_gaze_min) * window_size[0])
            pygame_y = int((-1*y - y_gaze_min) / (y_gaze_max - y_gaze_min) * window_size[1])

            dot_position = [pygame_x, pygame_y]
            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, dot_color, dot_position, dot_radius)

            # Draw thin grey lines for x and y axes
            pygame.draw.line(screen, grey, (0, window_size[1] // 2), (window_size[0], window_size[1] // 2), 1)
            pygame.draw.line(screen, grey, (window_size[0] // 2, 0), (window_size[0] // 2, window_size[1]), 1) 
            
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_color = (0, 0, 255)
            cv2.putText(frame, "No face detected", (10, 30), font, font_scale, text_color, font_thickness)
            screen.fill((255, 255, 255))

            no_gaze_text = "No gaze detection"
            font = pygame.font.Font(None, 36)
            text = font.render(no_gaze_text, True, (255, 0, 0))  # Red text
            text_rect = text.get_rect(center=(window_size[0] // 2, window_size[1] // 2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

        headpose_est(frame)

        cv2.imshow("Gaze point", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Plotting
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,9))

    plt.subplot(211)
    plt.step(range(len(dxt)), dxt, label='$\dot x$')
    plt.step(range(len(dxt)), dyt, label='$\dot y$')
    plt.title('Velocity Estimate')
    plt.legend(loc='best')
    plt.ylabel('Velocity')

    plt.subplot(212)
    plt.step(range(len(xt)), xt, label='$x$')
    plt.step(range(len(yt)), yt, label='$y$')
    plt.xlabel('Filter Step')
    plt.ylabel('Position')
    plt.legend(loc='best')
    plt.title('Position Estimate')
    plt.savefig('estimations.png')
    plt.show()


    plt.scatter(x_measured, y_measured, label='Measured', marker='o')  # Use 'o' marker for measured points
    plt.scatter(xt, yt, label='Kalman estimated', color='red', marker='x')  # Use 'x' marker for estimated points
    plt.scatter(x_gaus, y_gaus, label='Gaussian', color='green', marker='p')
    # plt.scatter(xt, yt, label='Position')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Measured and Estimated Gaze Positions')
    plt.legend()

    plt.savefig('difference.png')
    plt.show()
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
parser.add_argument('id', type=str, help='Are you using webcam or phone')
args = parser.parse_args()

if args.id == '1' or args.id == '0' or args.id == '2':
    id = int(args.id)
    print(id, type(id))
else:
    id = args.id
    print(id)

test_webcam(model, id)
