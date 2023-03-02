import json
import numpy as np
import os 

import torch 
from torch.utils.data import Dataset, DataLoader
import random
from utils import padding_idx

def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')  # encoding='latin1' to handle the inconsistency between python 2 and 3
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))
        feats = feats[inds]
    assert feats.shape[0] == max_length
    return np.float32(feats)

class kesvi_dataset(Dataset):
    def __init__(self, data_dir, file_path, split_type, tokenizers, max_vid_len, pair):
        src, tgt = pair
        maps = {'en':'en', 'ko':'ko'}
        self.data_dir = data_dir
        # load tokenizer
        self.tok_src, self.tok_tgt = tokenizers
        self.max_vid_len = max_vid_len
        self.split_type = split_type

        with open(self.data_dir+file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.srccaps, self.tgtcaps = [], []
        self.sent_ids = list(data.keys())
        for i in range(len(list(data.values()))):
            self.srccaps.append(list(data.values())[i][maps['ko']])
            self.tgtcaps.append(list(data.values())[i][maps['en']])

    def __len__(self):
        return len(self.srccaps)
        # return 64

    def __getitem__(self, idx):
        str_srccap,  sent_id = self.srccaps[idx], self.sent_ids[idx]
        vid = sent_id[:-2]
        srccap, srccap_mask, caplen_src = self.tok_src.encode_sentence(str_srccap)
        srcref = self.tok_src.encode_sentence_nopad_2str(str_srccap)

        try :
            s_video_feature = load_video_features(os.path.join(self.data_dir, 'video_features', 'scene_node', vid + '.npy'), self.max_vid_len) 
            s_video_graph = load_video_features(os.path.join(self.data_dir, 'video_features', 'scene_v_graph', vid + '.npy'), self.max_vid_len)  
        except FileNotFoundError :
            s_video_feature = load_fake_scene_node()
            s_video_graph = load_fake_scene_graph()

        if self.split_type != 'test':
            str_tgtcap = self.tgtcaps[idx]
            tgtcap, tgt_mask, caplen_tgt = self.tok_tgt.encode_sentence(str_tgtcap, flag='TRG')
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)
            return srccap, srccap_mask, (s_video_graph, s_video_feature), tgtcap, tgt_mask, caplen_tgt, srcref, tgtref
        else:
            str_tgtcap = self.tgtcaps[idx]
            tgtcap, _, _ = self.tok_tgt.encode_sentence(str_tgtcap, flag='TRG')
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)
            return srccap, srccap_mask, (s_video_graph, s_video_feature), sent_id, tgtcap[1:], srcref, tgtref


def get_loader(data_dir, tokenizers, split_type, batch_size, max_vid_len, pair, num_workers, pin_memory):
    maps = {'train':'kesvi_train.json', 'val': 'kesvi_val.json',
        'test': 'kesvi_test.json'}
    file_path = maps[split_type]
    mydata = kesvi_dataset(data_dir, file_path, split_type, tokenizers, max_vid_len, pair)
    if split_type in ['train']:
        shuffle = True
    elif split_type in ['val', 'test']:
        shuffle = False
    myloader = DataLoader(dataset=mydata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return myloader


def create_split_loaders(data_dir, tokenizers, batch_size, max_vid_len, pair, num_workers=0, pin_memory=False):
    train_loader = get_loader(data_dir, tokenizers, 'train', batch_size, max_vid_len, pair, num_workers, pin_memory)
    val_loader = get_loader(data_dir, tokenizers, 'val', batch_size, max_vid_len, pair, num_workers, pin_memory)
    test_loader = get_loader(data_dir, tokenizers, 'test', 1, max_vid_len, pair, num_workers, pin_memory)

    return train_loader, val_loader, test_loader 