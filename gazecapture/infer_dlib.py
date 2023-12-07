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
import dlib
import cv2

from ITrackerModel import ITrackerModel

CHECKPOINTS_PATH = 'C:/Users/Anja/kod/fixing-gazecapture-master'

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def getFaceGrid(frameW, frameH, faceWidth, faceLeft, faceTop, faceHeight):
    gridW = 25
    gridH = 25
    scaleX = gridW / frameW
    scaleY = gridH / frameH

    grid = np.zeros((gridH, gridW))

    xLo = round(faceLeft * scaleX) + 1
    yLo = round(faceTop * scaleY) + 1
    w = round(faceWidth * scaleX)
    h = round(faceHeight * scaleY)
    xHi = xLo + w - 1
    yHi = yLo + h - 1

    xLo = min(gridW, max(1, xLo))
    xHi = min(gridW, max(1, xHi))
    yLo = min(gridH, max(1, yLo))
    yHi = min(gridH, max(1, yHi))

    grid[yLo: yHi, xLo: xHi] = np.ones((yHi - yLo, xHi - xLo))

    grid = np.asmatrix(grid)
    grid = grid.getH()
    grid = grid[:].getH()
    labelFaceGrid = grid
    return labelFaceGrid

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_eye(shape, i, j, image, ):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    return roi

class SubtractMean(object):

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        return tensor.sub(self.meanImg)

one_size = 224
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Anja\Desktop\shape_predictor_68_face_landmarks.dat")
faceMean = sio.loadmat('mean_face_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']
eyeLeftMean = sio.loadmat('mean_left_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']
eyeRightMean = sio.loadmat('mean_right_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']

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

def get_data_from_webcam(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if (len(rects) == 0):
        return None
    left_eye_img = []
    right_eye_img = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        got_left = False
        got_right = False
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if (got_right and got_left):
                break
            if(name == 'right_eye'):
                right_eye_img = get_eye(shape, i, j, image)
                got_right = True
            elif(name == 'left_eye'):
                left_eye_img = get_eye(shape, i, j, image)
                got_left = True
    x = rects[0].left()
    y = rects[0].top()
    w = rects[0].right() - x
    h = rects[0].bottom() - y
    roi_face = image[y:y + h, x:x + w]
    face_mask = getFaceGrid(image.shape[1], image.shape[0], w, x, y, h)

    roi_face = transformFace(Image.fromarray(roi_face))
    left_eye_img = transformEyeL(Image.fromarray(left_eye_img))
    right_eye_img = transformEyeR(Image.fromarray(right_eye_img))
    face_mask = np.expand_dims(face_mask, 0)
    return left_eye_img, right_eye_img, roi_face, face_mask, [[100, 100]]


def test_webcam(model):
    video_capture = cv2.VideoCapture(0)
    center = [int(one_size/2), int(one_size/2)]
    while (True):
        _, frame = video_capture.read()
        frame = imutils.resize(frame, width=500)
        data_from_frame = get_data_from_webcam(frame)
        if (data_from_frame is None):
            continue
        eye_left, eye_right, face, face_mask, y = data_from_frame
        eye_left_t = torch.unsqueeze(eye_left,0)
        eye_right_t = torch.unsqueeze(eye_right, 0)
        face_t = torch.unsqueeze(face, 0)
        face_mask_t = torch.FloatTensor(face_mask)

        output = model(face_t, eye_left_t, eye_right_t, face_mask_t)
        predicted_pose = output.detach().cpu().numpy()
        frame = cv2.resize(frame, (one_size, one_size))
        frame = cv2.circle(frame, (center[0] + int(predicted_pose[0][0]), center[1] + int(predicted_pose[0][1])), 5, (0, 255, 0), -1)
        print(predicted_pose)
        cv2.imshow("gaze point", frame)
        cv2.waitKey(1)

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

test_webcam(model)