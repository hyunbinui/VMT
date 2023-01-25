import os
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
from gluoncv.utils.filesystem import try_import_decord

if not os.path.exists('/data/VMT/action_features/'):
    os.makedirs('/data/VMT/action_features/')

decord = try_import_decord()

model_name = 'i3d_inceptionv1_kinetics400'
net = get_model(model_name, nclass=400, pretrained=True)
print('%s model is successfully loaded.' % model_name)

directory_path = "/data/VMT/video_data/"
# directory_path = "/home/ubuntu/workspace/221122_vmt_hyunbin/DEAR/action_feature_extraction/code_kunst/"
for file in os.listdir(directory_path):
    filename = os.fsdecode(file)
    # video_fname = '/home/ubuntu/workspace/221122_vmt_hyunbin/workspace/code_kunst.mp4'
    vr = decord.VideoReader(directory_path+filename)
    frame_id_list = range(0, 64, 2)
    try:    
        video_data = vr.get_batch(frame_id_list).asnumpy()
        print(video_data.shape)
        clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

        transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        clip_input = transform_fn(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
        print(clip_input.shape)
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        print(clip_input.shape)
        # print('Video data is downloaded and preprocessed.')

        pred = net(nd.array(clip_input)).asnumpy()
        np.save('/home/ubuntu/workspace/221122_vmt_hyunbin/DEAR/action_feature_extraction/feature/'+f'{filename[:-4]}'+'.npy', pred)
        print(filename)
    except IndexError:
        pass

# (120, 720, 1280, 3)
# (1, 1, 120, 720, 1280, 3)
# load time:  12.359415292739868
# (1, 1, 120, 720, 1280, 3)