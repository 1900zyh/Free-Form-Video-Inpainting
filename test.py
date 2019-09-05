# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import datetime
import random
import sys

### My libs
from models.dataset import dataset
from models.CPNet_model import CPNet
 

parser = argparse.ArgumentParser(description="CPNet")
parser.add_argument("-b", type=int, default=1)
parser.add_argument("-e", type=int, default=0)
parser.add_argument("-n", type=str, default='youtube-vos') 
parser.add_argument("-m", type=str, default='fixed') 
args = parser.parse_args()

BATCH_SIZE = args.b
RESUME = args.e
DATA_NAME = args.n
MASK_TYPE = args.m

w,h = 424, 240
default_fps = 6


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def get_clear_state_dict(old_state_dict):
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = v
  return new_state_dict


def main_worker(gpu, ngpus_per_node, args):
  if ngpus_per_node > 0:
    torch.cuda.set_device(int(gpu))
  # set random seed 
  seed = 2020
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  # Model and version
  model = set_device(CPNet())
  data = torch.load('./weight/weight.pth', map_location = lambda storage, loc: set_device(storage))
  data = get_clear_state_dict(data)
  model.load_state_dict(data)
  model.eval() # turn-off BN

  Pset = dataset(DATA_NAME, MASK_TYPE)
  step = math.ceil(len(Pset) / ngpus_per_node)
  Pset = torch.utils.data.Subset(Pset, range(gpu*step, min(gpu*step+step, len(Pset))))
  Trainloader = torch.utils.data.DataLoader(Pset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

  num_length = 120 # 2*num_length is the number of reference frames
  save_path = 'results/{}_{}'.format(DATA_NAME, MASK_TYPE)
  for vi, V in enumerate(Trainloader):
    frames, masks, GTs, info = V # b,3,t,h,w / b,1,t,h,w
    frames, masks, GTs = set_device([frames, masks, GTs])
    seq_name = info['name'][0]
    num_frames = frames.size()[2]
    print('[{}] {}/{}: {} for {} frames ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
      vi, len(Trainloader), seq_name, frames.size()[2]))

    with torch.no_grad():
      rfeats = model(frames, masks)
    frames_ = frames.clone()
    masks_ = masks.clone() 
    index = [f for f in reversed(range(num_frames))]
    comp_frames = []
    pred_frames = []
    mask_frames = []
    orig_frames = []
        
    for t in range(2): # forward : 0, backward : 1
      if t == 1:
        comp0 = frames.clone()
        frames = frames_
        masks = masks_
        index.reverse()

      for f in index:
        ridx = []
        start = f - num_length
        end = f + num_length
        if f - num_length < 0:
          end = (f + num_length) - (f - num_length)
          if end > num_frames:
            end = num_frames -1
          start = 0
        elif f + num_length > num_frames:
          start = (f - num_length) - (f + num_length - num_frames)
          if start < 0:
            start = 0
          end = num_frames -1
            
        # interval: 2
        for i in range(start, end, 2):
          if i != f:
            ridx.append(i)
        
        with torch.no_grad():
          comp, pred, masked, orig = model(rfeats[:,:,ridx], frames[:,:,ridx], masks[:,:,ridx], frames[:,:,f], masks[:,:,f], GTs[:,:,f])
          c_s = comp.shape
          Fs = torch.empty((c_s[0], c_s[1], 1, c_s[2], c_s[3])).float().cuda()
          Hs = torch.zeros((c_s[0], 1, 1, c_s[2], c_s[3])).float().cuda()
          Fs[:,:,0] = comp.detach()
          frames[:,:,f] = Fs[:,:,0]
          masks[:,:,f] = Hs[:,:,0]                
          rfeats[:,:,f] = model(Fs, Hs)[:,:,0]

        if t == 1:
          est = comp0[:,:,f] * (len(index)-f) / len(index) + comp.detach() * f / len(index)
          comp = (est[0].cpu().permute(1,2,0).numpy() * 255.).astype(np.uint8)
          pred = (pred[0].cpu().permute(1,2,0).numpy() * 255.).astype(np.uint8)
          masked = (masked[0].cpu().permute(1,2,0).numpy() * 255.).astype(np.uint8)
          orig = (orig[0].cpu().permute(1,2,0).numpy() * 255.).astype(np.uint8)
          if comp.shape[1] % 2 != 0:
            comp = np.pad(comp, [[0,0],[0,1],[0,0]], mode='constant')
            pred = np.pad(pred, [[0,0],[0,1],[0,0]], mode='constant')
            masked = np.pad(masked, [[0,0],[0,1],[0,0]], mode='constant')
            orig = np.pad(orig, [[0,0],[0,1],[0,0]], mode='constant')
          comp_frames.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
          pred_frames.append(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
          mask_frames.append(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
          orig_frames.append(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

    os.makedirs(os.path.join(save_path, seq_name), exist_ok=True)
    comp_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'comp.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    pred_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'pred.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    mask_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'mask.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    orig_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'orig.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    for f in range(len(comp_frames)):
      comp_writer.write(comp_frames[f])
      pred_writer.write(pred_frames[f])
      mask_writer.write(mask_frames[f])
      orig_writer.write(orig_frames[f])
    comp_writer.release()
    pred_writer.release()
    mask_writer.release()
    orig_writer.release()

  print('Finish in {}'.format(save_path))



if __name__ == '__main__':
  config = json.load(open('configs/config.json'))
  ngpus_per_node = torch.cuda.device_count()
  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  if ngpus_per_node > 0:
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    main_worker(0, 1, args)
