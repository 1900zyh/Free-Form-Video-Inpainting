import os
import cv2 
import torch
from glob import glob 
from tqdm import tqdm 
import numpy as np 
from cv2 import VideoWriter_fourcc

from model.video_inpainting_model import VideoInpaintingModel


fps = 30 
DATA_PATH = '/data02/t-yazen/datasets/msrvtt'
SAVE_PATH = '/data02/t-yazen/vi-baselines/results/ffvi'
video_path = 'eval-clip100'
mask_path = 'eval-clip100-curve-mask'
os.makedirs(SAVE_PATH, exist_ok=True)


def to_img(x):
    tmp = x.cpu().data.numpy().transpose((1,2,0))
    tmp = np.clip(tmp,0,1)*255.
    return tmp.astype(np.uint8)


def load_video(vpath):
    frame_cap = cv2.VideoCapture(vpath)
    mask_cap = cv2.VideoCapture(vpath.replace(video_path, mask_path).replace('.mp4', '.avi'))
    
    images, masks = [], []
    original_images, original_masks = [], []
    _, frame = frame_cap.read()
    _, mask = mask_cap.read()
    while frame is not None: 
        original_images.append(frame.copy())
        frame = np.float32(frame)/255.0
        images.append(torch.from_numpy(frame))

        mask = (mask == 255).astype(np.float32)
        original_masks.append(mask.copy())
        mask = mask[:,:,0]
        select_value = 0
        mask = (mask == select_value).astype(np.float32)
        masks.append( torch.from_numpy(mask) )

        _, frame = frame_cap.read()
        _, mask = mask_cap.read()

    masks = torch.stack(masks).float().unsqueeze(1) # [T, C, H, W]
    images = torch.stack(images).float().permute(0, 3, 1, 2)   #[T, C, H, W]
    
    # extend dimension for batch size
    masks = masks.unsqueeze(0)
    inputs = images.unsqueeze(0)
    return masks, inputs, original_images, original_masks



def main_worker(): 

    # load model
    data = torch.load('../weights/release.pth', map_location='cpu')
    config = data['config']['arch']['args']
    model = VideoInpaintingModel(**config)
    model.load_state_dict(data['state_dict'])
    model = model.eval().cuda()
    
    
    # load data list 
    video_list = sorted(glob(os.path.join(DATA_PATH, video_path, '*.mp4')))
    video_list = video_list

    for vpath in tqdm(video_list): 
        masks, inputs, original_images, original_masks = load_video(vpath)
        batch, frame, channel, height, width = inputs.size()
        fourcc = VideoWriter_fourcc(*'mp4v')
        vpath = os.path.join(SAVE_PATH, os.path.basename(vpath))
        videoWriter = cv2.VideoWriter(vpath, fourcc, fps, (width, height))

        per_frame = 30 
        # imgs: [B, L, C=3, H, W]
        # masks: [B, L, C=1, H, W]
        # guidances: [B, L, C=1, H, W]
        for i in range(0, frame, per_frame): 
            with torch.no_grad():
                i_ = inputs[:,i:min(i+per_frame, frame)].cuda()
                m_ = masks[:,i:min(i+per_frame, frame)].cuda() 
                outputs = model(i_, m_)['outputs']
                outputs = outputs.clamp(0, 1)
    
            for j in range(outputs.size(1)): 
                out_frame = to_img(outputs[0, j, :,:,:])
                out_frame = out_frame * original_masks[i+j] + original_images[i+j] * (1 - original_masks[i+j])
                out_frame = out_frame.astype(np.uint8)
                videoWriter.write(out_frame)
        videoWriter.release()
            
            
if __name__ == '__main__': 
    
    main_worker()
