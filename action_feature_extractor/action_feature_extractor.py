from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
import json
import math

import numpy as np
import torch
import imageio.v3 as iio
import time
from tqdm import tqdm 

from i3dpt import I3D


def get_features(sample, model):
    sample = sample.transpose(0, 4, 1, 2, 3)
    with torch.no_grad():
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        out_var = model.extract(sample_var)
        out_tensor = out_var.data.cpu()
        return out_tensor.numpy()


def read_video(vid):
    # start = time.time()
    data = []

    I = iio.imread(vid)
    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)
    I = (I.astype('float32') / 255.0 - 0.5) * 2
    data.append(I)

    if len(data) <= 0:
        return None
    res = np.asarray(data)[:, :, :, :]
    # print("load time: ", time.time() - start)
    return res


def run(args):

    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.cuda()

    for file in tqdm(os.listdir(args.video_dir)):
        filename = os.fsdecode(file)[:-4]
         
        if os.path.exists(os.path.join(args.out_dir, filename +'.npy')):
            continue

        video = os.path.join(args.video_dir, file)
        clip = read_video(video)

        if clip is None:
            continue

        clip_len = clip.shape[1]

        if clip_len <= args.max_interval:
            features = get_features(clip, i3d_rgb)
        else:
            tmp_1 = 0
            features = []
            while True:
                tmp_2 = tmp_1 + args.max_interval
                tmp_2 = min(tmp_2, clip_len)
                feat = get_features(clip[:, tmp_1:tmp_2], i3d_rgb)
                features.append(feat)
                if tmp_2 == clip_len:
                    break
                tmp_1 = max(0, tmp_2 - args.overlap)

            features = np.concatenate(features, axis=1)

        np.save(os.path.join(args.out_dir, filename + '.npy'), features)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_interval', type=int, default=120, help='max interval of video clip')
    parser.add_argument('--overlap', type=int, default=25, help='overlap')

    parser.add_argument('--rgb_weights_path', type=str, default='./VMT-all-at-once/action_feature_extractor/model_rgb.pth', help='Path to rgb model state_dict')
    parser.add_argument('--video_dir', type=str, default='./dataset/video_data', help='directory that contains video clips')
    parser.add_argument('--out_dir', type=str, default='./dataset/action_features/', help='directory for extracted action features')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    run(args)
