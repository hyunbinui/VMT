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

from i3dpt import I3D

FPS = 25
MAX_INTERVAL = 400
OVERLAP = 25

parser = argparse.ArgumentParser()

parser.add_argument('--rgb_weights_path', type=str, default='/home/ubuntu/workspace/221122_vmt_hyunbin/DEAR/action_feature_extraction/model_rgb.pth', help='Path to rgb model state_dict')
parser.add_argument('--video_dir', type=str, default='/home/ubuntu/workspace/221122_vmt_hyunbin/DEAR/action_feature_extraction/code_kunst', help='directory that contains video clips')
parser.add_argument('--out_dir', type=str, default='/home/ubuntu/workspace/221122_vmt_hyunbin/DEAR/action_feature_extraction/code_kunst/feature', help='directory for extracted action features')

args = parser.parse_args()


def get_features(sample, model):
    sample = sample.transpose(0, 4, 1, 2, 3)

    with torch.no_grad():
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        # sample_var = torch.autograd.Variable(torch.from_numpy(sample).cpu(), volatile=True)
        out_var = model.extract(sample_var)
        out_tensor = out_var.data.cpu()
        return out_tensor.numpy()


def read_video(video_dir):
    # start = time.time()
    frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    data = []

    for i, frame in enumerate(sorted(frames)):
        I = iio.imread(os.path.join(video_dir, frame))

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I = (I.astype('float32') / 255.0 - 0.5) * 2
        data.append(I)

    if len(data) <= 0:
        return None
    print(I.shape)
    print(data)
    res = np.asarray(data)[:, :, :, :]
    # print("load time: ", time.time() - start)
    return res


def run(args):
    # Run RGB model
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.cuda()

    for file in os.listdir(args.video_dir):
        filename = os.fsdecode(file)[:-4]

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        if os.path.exists(os.path.join(args.out_dir, filename +'.npy')):
            continue

        video = os.path.join(args.video_dir)
        clip = read_video(video)

        if clip is None:
            continue

        clip_len = clip.shape[1]
        if clip_len <= MAX_INTERVAL:
            features = get_features(clip, i3d_rgb)
        else:
            tmp_1 = 0
            features = []
            while True:
                tmp_2 = tmp_1 + MAX_INTERVAL
                tmp_2 = min(tmp_2, clip_len)
                feat = get_features(clip[:, tmp_1:tmp_2], i3d_rgb)
                features.append(feat)
                if tmp_2 == clip_len:
                    break
                tmp_1 = max(0, tmp_2 - OVERLAP)
            features = np.concatenate(features, axis=1)

        np.save(os.path.join(args.out_dir, filename + '.npy'), features)


if __name__ == "__main__":
    run(args)
