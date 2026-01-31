import os
import numpy as np
import torch
import json
from utils import *

class Jamendo_Dataset(object):
    '''
    Jamendo dataset
    '''
    def __init__(self, root_path, is_incontext = False, codec_layer = 4, L = 2588) -> None:
        '''
        '''
        self.root = root_path
        self.audio_token_path = os.path.join(self.root, 'segments_no_vocals_dac')
        # self.meta_path = os.path.join(self.root, 'jamendo_metadata.json')
        self.meta_path = os.path.join(self.root, 'meta_without_SDD_small.json')
        # self.meta_path = os.path.join(self.root, 'electronic_1000.json')
        
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.meta_key = list(self.meta.keys())
        
        self.codec_layer = codec_layer
        self.length = L
        self.special_token_id = 1024
        self.is_incontext = is_incontext
    
    def __getitem__(self, idx):
        path = os.path.join(self.audio_token_path, self.meta[self.meta_key[idx]]['path'].replace('.mp3', '.npy'))
        # path = os.path.join(self.audio_token_path, self.meta[idx]['path'].replace('.mp3', '.npy'))
        description = self.meta[self.meta_key[idx]]['rephrased_caption']
        data = np.load(path, allow_pickle=True)
        data = data[:self.codec_layer, :]
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        data = to_delay(data)
        
        mask = (data == self.special_token_id)
        if self.is_incontext:
            return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, :]), description

        return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta_key)

class MB_Dataset(object):
    '''
    MusicBench dataset
    '''
    def __init__(self, metadata, root_path, codec_layer = 4, L = 500) -> None:
        self.meta = metadata
        self.root = root_path
        self.codec_layer = codec_layer
        self.length = L
        self.number = len(self.meta)
        self.special_token_id = 2048
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.meta[idx]['location'][:-4]+'.npy')
        description = self.meta[idx]['main_caption']
        data = np.load(path, allow_pickle=True)
        data = data[:self.codec_layer, :]
        K, L = data.shape
        # print(K, L)
        # data = torch.LongTensor(np.pad(data,((0, 0), (self.length-L, 0)),'constant',constant_values=(0,0)))
        if L >= self.length:
            data = data[:, :self.length]
        else:
            data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)), 'constant', constant_values=(0, self.special_token_id)))
        data = to_delay(data)
        # data = torch.LongTensor(data)
        # mask = torch.ne(torch.LongTensor(data), 2048)
        # return seq[:, 0:499], seq[:, 1:500]
        mask = (data == 2048)
        return torch.LongTensor(data[:, :-1]), mask[0], torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta)

if __name__ == '__main__':
    # test code pattern
    train_data = Jamendo_Dataset(root_path = './datasets/Jamendo')
    print(len(train_data))
    print('Done')
    
    # test