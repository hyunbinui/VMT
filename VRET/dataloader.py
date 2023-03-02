import json
import numpy as np
import pandas as pd
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch 
from torch.utils.data import Dataset, DataLoader
import random
from utils import padding_idx


def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')  # encoding='latin1' to handle the inconsistency between python 2 and 3
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0), (0, 0)), 'constant', constant_values=0) 
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))   
        feats = feats[inds]
    assert feats.shape[0] == max_length
    img = torch.from_numpy(np.float32(feats))  # torch.Size([32, 1024])

    # img_mask = (torch.sum(img != 1, dim=1) != 0).unsqueeze(-2)  # torch.Size([1, 32])
    # img = img * img_mask.squeeze().unsqueeze(-1).expand_as(img).float()  # torch.Size([32, 1024])

    return img

class MSVD_dataset(Dataset):
    def __init__(self, data_dir, file_path, split_type, tokenizers, max_vid_len, pair):
        src, tgt = pair
        maps = {'en':'en', 'tr':'tr'}
        self.data_dir = data_dir
        # load tokenizer
        self.tok_src, self.tok_tgt = tokenizers
        self.max_vid_len = max_vid_len
        self.split_type = split_type

        df = pd.read_csv(self.data_dir+'label/'+file_path)

        self.srccaps = df[maps['tr']].tolist()
        self.tgtcaps = df[maps['en']].tolist()
        self.sent_ids = df['vid_id'].tolist()

    def __len__(self):
        return len(self.srccaps)
        # return 64

    def __getitem__(self, idx):
        str_srccap,  sent_id = self.srccaps[idx], self.sent_ids[idx]
        vid = sent_id.split('@')[0]
        srccap, srccap_mask, caplen_src = self.tok_src.encode_sentence(str_srccap)
        srcref = self.tok_src.encode_sentence_nopad_2str(str_srccap)

        s_video_feature = load_video_features(os.path.join(self.data_dir, 'video_features', 'scene_node', vid + '.npy'), self.max_vid_len) 
        s_video_graph = load_video_features(os.path.join(self.data_dir, 'video_features', 'scene_v_graph', vid + '.npy'), self.max_vid_len)  

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
    maps = {'train':'train.csv', 'val': 'val.csv',
        'test': 'test.csv'}
    file_path = maps[split_type]
    mydata = MSVD_dataset(data_dir, file_path, split_type, tokenizers, max_vid_len, pair)
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
    # test_loader = [0]

    return train_loader, val_loader, test_loader