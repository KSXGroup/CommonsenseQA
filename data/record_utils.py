import torch
import pickle
import numpy as np
from typing import List
from .create_record_data import InputFeatures
from torch.utils.data.dataset import Dataset

class QADataset(Dataset):
    def __init__(self, feature_list:List[InputFeatures], seq_len:int):
        self.length = len(feature_list)
        self.tokens = np.zeros((self.length, seq_len), dtype=np.int32)
        self.masks =  np.zeros((self.length, seq_len), dtype=np.bool)
        self.seg_ids = np.zeros((self.length, seq_len), dtype=np.bool)
        self.start_positions = np.zeros((self.length,), dtype=np.int32)
        self.end_positions = np.zeros((self.length, ), dtype=np.int32)
        self.example_index = np.zeros((self.length, ), dtype=np.int32)

        print("loading %d features" % self.length)

        for i, feature in enumerate(feature_list):
            print("\rloading %d\t" % i, end='')
            self.tokens[i] = feature.input_ids
            self.masks[i] = feature.input_mask
            self.seg_ids[i] = feature.segment_ids
            self.start_positions[i] = feature.start_position
            self.end_positions[i] = feature.end_position
            self.example_index[i] = feature.example_index
        print()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (torch.tensor(self.tokens[item],dtype=torch.int64),
                torch.tensor(self.masks[item],dtype=torch.float32),
                torch.tensor(self.seg_ids[item], dtype=torch.int64),
                torch.tensor(self.start_positions[item], dtype=torch.int64),
                torch.tensor(self.end_positions[item], dtype=torch.int64),
                self.example_index[item])

class RawResult:
    def __init__(self, unique_id:int, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits


def load_record_dataset(path:str, max_len:int, train_prop:float) -> (QADataset,QADataset):
    with open(path, "rb") as f:
        feature_list = pickle.load(f)
    dlen = int(len(feature_list) * train_prop)
    return QADataset(feature_list[:dlen], max_len), QADataset(feature_list[dlen:], max_len)

def load_record_devset(path:str, max_len:int):
    with open(path, "rb") as f:
        feature_list = pickle.load(f)
    return QADataset(feature_list, max_len), feature_list