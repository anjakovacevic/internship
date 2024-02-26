import os
import numpy as np
import scipy.io as sio

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import mediapipe as mp

from ITrackerModel import ITrackerModel

CHECKPOINTS_PATH = '/home/anja/ros2_ws/src/system/system/gaze/checkpoint.pth.tar'

class SubtractMean(object):
    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        return tensor.sub(self.meanImg)
    

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

