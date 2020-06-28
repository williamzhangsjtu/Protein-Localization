import torch 
from imblearn.over_sampling import RandomOverSampler
from pypeln import process as pr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import h5py
import utils


class single_instance(Dataset):
    """
    label_dict: {Exxxxx: 0, Exxxxx: 1}
    data_h5[index]: T x 256 x 256 x 3
    """
    
    def __init__(self, data_h5, label_dict, transform, T=1):

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

        #img = self.transform(image=self.data[str(index)][idx-startfrom])['image']
        #img = torch.as_tensor(np.rollaxis(img, 2, 0))
        img = np.rollaxis(self.data[str(index)][idx-startfrom], 2, 0)  # H x W x 3

        # r, g, b = img[0], img[1], img[2]
        # #p = round(g.sum() / (r.sum() + g.sum() + b.sum()), 4)
        # tmp = g.copy()
        # g_tot = g.sum()
        # trh = 25
        # tmp[g > trh] = 1
        # tmp[g < trh] = 0
        # sum = tmp.sum() + 0.1
        # weight = np.log(g_tot / sum + 1)

        img = self.transform(torch.as_tensor(img))
        
        return img, self.labels[index], 1, index #, torch.as_tensor(weight)


    def __len__(self):
        return len(self.num2idx)






def dataloader_single(data_h5, label_dict, transform, T=1, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("num_workers", 8)
    kwargs.setdefault("shuffle", True)



    _dataset = single_instance(data_h5, label_dict, transform, T=1)

    return DataLoader(_dataset, **kwargs)




class oversample_dataset(Dataset):
    """
    label_dict: {Exxxxx: 0, Exxxxx: 1}
    data_h5[index]: T x 256 x 256 x 3
    """
    
    def __init__(self, data_h5, indexes, labels, transform):

        self.data = None
        self.data_h5 = data_h5
        self.transform = transform
        self.indexes = indexes
        self.labels = labels
        

        

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index, place = self.indexes[idx].split('_')

        img = np.rollaxis(self.data[str(index)][int(place)], 2, 0)  # 3 x H x W

        # r, g, b = img[0], img[1], img[2]
        # #p = round(g.sum() / (r.sum() + g.sum() + b.sum()), 4)
        # tmp = g.copy()
        # g_tot = g.sum()
        # trh = 25
        # tmp[g > trh] = 1
        # tmp[g < trh] = 0
        # sum = tmp.sum() + 0.1
        # weight = np.log(g_tot / sum + 1)

        img = self.transform(torch.as_tensor(img))

        
        return img, self.labels[index], 1, index #, torch.as_tensor(weight)
        


    def __len__(self):
        return len(self.indexes)


def oversample_dataloader(data_h5, label_dict, transform, T=1, **kwargs):
    ros = RandomOverSampler()
    indexes, label = zip(*label_dict.items())
    index_place, index_place_resample = [], []
    labels, label_resample = [], []
    with h5py.File(data_h5, 'r') as data:
        for i, index in enumerate(indexes):
            for j in range(len(data[index])):
                index_place.append('{}_{}'.format(index, j))
                labels.append(label[i])
    for i, label in enumerate(labels):
        for l in torch.where(label == 1)[0]:
            label_resample.append(l)
            index_place_resample.append(index_place[i])
    resample_index, label_resample = ros.fit_resample(np.arange(len(label_resample)).reshape(-1, 1), label_resample)
    index_place_resample = [index_place_resample[i[0]] for i in resample_index]
    _dataset = oversample_dataset(data_h5, index_place_resample, label_dict, transform)
    return DataLoader(_dataset, **kwargs)
