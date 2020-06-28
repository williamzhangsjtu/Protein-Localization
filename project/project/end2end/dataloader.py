import torch 
from pypeln import process as pr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import h5py
import utils


class pretrain_dataset(Dataset):
    """
    label_dict: {Exxxxx: 0, Exxxxx: 1}
    data_h5[index]: T x 256 x 256 x 3
    """
    
    def __init__(self, data_h5, label_dict, transform):

        self.data = None
        self.data_h5 = data_h5
        self.transform = transform
        self.labels = label_dict
        self.idxstart = {}
        self.num2idx = {}
        with h5py.File(data_h5, 'r') as f:
            startfrom = 0
            for key in label_dict:
                self.idxstart[key] = startfrom
                for i in range(startfrom, startfrom + len(f[key])):
                    self.num2idx[i] = key
                startfrom += len(f[key])
        

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.num2idx[idx]
        startfrom = self.idxstart[index]

        img = np.rollaxis(self.data[str(index)][idx-startfrom], 2, 0)  # H x W x 3
        img = self.transform(torch.as_tensor(img))
        
        return img, self.labels[index], index


    def __len__(self):
        return len(self.num2idx)


class folder_dataset(Dataset):
    """
    data_h5: {name: feature (T x D)}
    label_dict: {name: label}
    """

    def __init__(self, data_h5, label_dict):

        self.data = None
        self.data_h5 = data_h5
        self.labels = label_dict
        self.idx2name = {i:name for i, name in enumerate(label_dict.keys())}

    def __getitem__(self, idx):
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')

        name = self.idx2name[idx]

        data = torch.as_tensor(self.data[str(name)][()])  # T x D

        return data, self.labels[name], len(data), name

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    batch.sort(key=lambda d: len(d[0]), reverse=True)
    data, label, length, name = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    return data, torch.stack(label), \
            torch.as_tensor(length, dtype=torch.long), name

def collate_FN(batch):
    data, label, name = zip(*batch)
    return torch.stack(data), torch.stack(label), name

def pretrain_dataloader(data_h5, label_dict, transform, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("num_workers", 8)
    kwargs.setdefault("shuffle", True)



    _dataset = pretrain_dataset(data_h5, label_dict, transform)

    return DataLoader(_dataset, collate_fn=collate_FN, **kwargs)



def predict_dataloader(data_h5, label_dict, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("num_workers", 8)
    kwargs.setdefault("shuffle", True)



    _dataset = folder_dataset(data_h5, label_dict)

    return DataLoader(_dataset, collate_fn=collate_fn, **kwargs)


