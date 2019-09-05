# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

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
import json

### My libs
from core.video_inpainting_model import VideoInpaintingModel
from core.transform import Stack, ToTorchFormatTensor
from core.util import set_device, postprocess, ZipReader, set_seed
 

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
sample_length = 7
default_fps = 6
ngpus = torch.cuda.device_count()
_to_tensors = transforms.Compose([
  Stack(),
  ToTorchFormatTensor()])


def get_clear_state_dict(old_state_dict):
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = set_device(v)
  return new_state_dict


def get_mask(vname, mask_dict, f):
  if MASK_TYPE == 'fixed':
    m = np.zeros((h,w), np.uint8)
    m[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 255
    return Image.fromarray(m)
  elif MASK_TYPE == 'object':
    mname = mask_dict[f]
    m = ZipReader.imread('../datazip/{}/Annotations/{}.zip'.format(DATA_NAME, vname), mname).convert('L')
    m = np.array(m)
    m = np.array(m>0).astype(np.uint8)*255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4)
    return Image.fromarray(m)
  else:
    raise NotImplementedError(f"Mask type {MASK_TYPE} not exists")



def main_worker(gpu, ngpus_per_node, args):
  if ngpus_per_node > 0:
    torch.cuda.set_device(int(gpu))
  # set random seed 
  set_seed(2020)

  # Model and version
  config = torch.load(args.resume)['config']
  model = VideoInpaintingModel(**config['arch']['args'])
  model = set_device(model)
  state_dict = torch.load(args.resume)['state_dict']
  state_dict = get_clear_state_dict(state_dict)
  model.load_state_dict(state_dict)
  model.eval() 

  # prepare dataset
  save_path = 'results/{}_{}'.format(DATA_NAME, MASK_TYPE)
  with open('../flist/{}/test.json'.format(DATA_NAME), 'r') as f:
    videos_dict = json.load(f)
  video_names = list(videos_dict.keys())
  with open('../flist/{}/mask.json'.format(DATA_NAME), 'r') as f:
    masks_dict = json.load(f)
  mask_names = list(masks_dict.keys())
  step = math.ceil(len(video_names) / ngpus_per_node)
  video_names = video_names[gpu*step: min(gpu*step+step, len(video_names))]
  mask_names = mask_names[gpu*step: min(gpu*step+step, len(mask_names))]
  # iteration through datasets
  for vi, vname in enumerate(video_names):
    index = 0
    fnames = videos_dict[vname]
    orig_video = []
    mask_video = []
    comp_video = []
    pred_video = []
    os.makedirs(os.path.join(save_path, vname), exist_ok=True)
    print('{}/{} to {} : {} of {} frames ...'.format(vi, len(video_names), save_path, vname, len(fnames)))
    while index < len(fnames):
      # preprocess data
      frames = []
      masks = []
      for f, fname in enumerate(fnames[index:min(len(fnames), index+sample_length)]):
        img = ZipReader.imread('../datazip/{}/JPEGImages/{}.zip'.format(DATA_NAME, vname), fname)
        img = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (w,h), cv2.INTER_CUBIC)
        frames.append(Image.fromarray(img))
        masks.append(get_mask(vname, masks_dict, f))
      if len(frames) < sample_length:
        frames += [frames[-1]] * (sample_length-len(frames))
        masks += [masks[-1]] * (sample_length-len(masks))
      # inference
      frames = _to_tensors(frames).unsqueeze(0)
      masks =  _to_tensors(masks).unsqueeze(0)
      frames, masks = set_device([frames, masks])
      with torch.no_grad():
        pred_img = model(frames, masks, model='G')['outputs']
      # postprocess
      complete_img = (pred_img * masks) + (frames * (1. - masks))
      masked_img = frames * (1. - masks) + masks
      orig_video.extend(postprocess(frames[0]))
      comp_video.extend(postprocess(complete_img[0]))
      pred_video.extend(postprocess(pred_img[0]))
      mask_video.extend(postprocess(masked_img[0]))
      # next clip
      index += sample_length
    # save all frames into a video
    writers = {tname: (cv2.VideoWriter(os.path.join(save_path, vname, '{}.avi'.format(tname)),
                          cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h)), frames_to_save)
              for tname,frames_to_save in zip(['orig', 'pred', 'mask', 'comp'], [orig_video, pred_video, mask_video, comp_video])}
    for wtype, (writer, imgs) in writers.items():
      for i in range(len(fnames)):
        writer.write(cv2.cvtColor(np.array(imgs[i]), cv2.COLOR_RGB2BGR))
      writer.release()

  print('Finish in {}'.format(save_path))



if __name__ == '__main__':
  args.resume = 'weights/release.pth'
  ngpus_per_node = torch.cuda.device_count()
  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  if ngpus_per_node > 0:
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    main_worker(0, 1, args)
