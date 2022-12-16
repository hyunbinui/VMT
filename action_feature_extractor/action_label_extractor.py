
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
from gluoncv.utils.filesystem import try_import_decord


def get_frame_id_list(vr):

    if len(vr) > 64:
        frame_id_list = range(0, 64, 2)
    else:
        frame_id_list = []
        frame_id = [*range(0,len(vr),2)]
        while len(frame_id_list) != 32 :
            for i in frame_id:
                frame_id_list.append(i)
                if len(frame_id_list) == 32:
                    break

    return frame_id_list


def preprocess_video_data(net, vr, frame_id_list):

    video_data = vr.get_batch(frame_id_list).asnumpy()
    clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    pred = net(nd.array(clip_input))

    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    index = []
    for i in range(topK):
        if nd.softmax(pred)[0][ind[i]].asscalar() > 0.01:
            index.append(int(ind[i].asscalar()))

    return index


def to_json(dict, output_path):
    with open(output_path+"action_label.json", "w", encoding='utf-8') as outfile:
        json.dump(dict, outfile, indent=2 , ensure_ascii=False)


def run(args):

    decord = try_import_decord()
    model_name = 'i3d_inceptionv1_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    print('%s model is successfully loaded.' % model_name)

    dict = {}

    for file in tqdm(os.listdir(args.video_dir)):
        filename = os.fsdecode(file)
        vr = decord.VideoReader(args.video_dir+filename)

        frame_id_list = get_frame_id_list(vr)
        index = preprocess_video_data(net, vr, frame_id_list)
        
        dict[filename[:-4]] = index

        to_json(dict, args.out_dir)

 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--video_dir', type=str, default='./dataset/video_data', help='directory that contains video clips')
    parser.add_argument('--out_dir', type=str, default='./dataset/', help='directory for extracted action features')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    run(args)
