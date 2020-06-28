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


class test_single_instance(Dataset):
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

        img = self.data[str(index)][idx-startfrom]  # 3 x H x W
        img = np.rollaxis(img, 2, 0)
        imgs = []
        for transform in self.transform:
            imgs.append(transform(torch.as_tensor(img))) 

        
        return imgs, self.labels[index], 1, index


    def __len__(self):
        return len(self.num2idx)
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
        img = self.transform(torch.as_tensor(img))
        
        return img, self.labels[index], 1, index


    def __len__(self):
        return len(self.num2idx)

class multiple_instance(Dataset):
    """
    images[i]: [Exxxxx/xxx.jpg, Exxxxx/xxx.jpg, ...] (T)
    """
    
    def __init__(self, data_h5, label_dict, transform, time_step=None):
        self.label_dict = label_dict
        self.data = None
        self.data_h5 = data_h5
        self.transform = transform
        self.time_step = time_step

        self.indexes = []
        self.range = []
        self.re_sample()
        

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.indexes[idx]
        pick = self.range[idx]
        data = None
        imgs = np.rollaxis(self.data[str(index)][pick], 3, 1) # T x 3 x H x W
        for img in imgs:
            img = self.transform(torch.as_tensor(img)).unsqueeze(0)
            data = torch.cat((data, img), dim=0) \
                if data is not None else img

        return data, self.label_dict[index], len(data), index


    def __len__(self):
        return len(self.indexes)


    def re_sample(self):
        self.indexes, self.range = [], []
        with h5py.File(self.data_h5, 'r') as f:
            for idx in self.label_dict:
                length = len(f[idx])
                if self.time_step is None:
                    self.indexes.append(idx)
                    self.range.append([0, length])
                else:
                    shuffle = np.arange(length)
                    np.random.shuffle(shuffle)
                    for i in range(length// self.time_step):
                        self.indexes.append(idx)
                        shuffle_idx = shuffle[i * self.time_step : (i + 1) * self.time_step]
                        shuffle_idx.sort()
                        self.range.append(shuffle_idx.tolist())
                    if (length// self.time_step) * self.time_step < length - 4:
                        self.indexes.append(idx)
                        shuffle_idx = shuffle[(length// self.time_step) * self.time_step : length]
                        shuffle_idx.sort()
                        self.range.append(shuffle_idx.tolist())





class multiple_instance_bak(Dataset):
    """
    images[i]: [Exxxxx/xxx.jpg, Exxxxx/xxx.jpg, ...] (T)
    """
    
    def __init__(self, data_h5, label_dict, transform, time_step=None):
        self.label_dict = label_dict
        self.data = None
        self.data_h5 = data_h5
        self.transform = transform

        self.indexes = []
        self.range = []
        with h5py.File(data_h5, 'r') as f:
            for idx in label_dict:
                length = len(f[idx])
                if time_step is None:
                    self.indexes.append(idx)
                    self.range.append([0, length])
                else:
                    for i in range(length// time_step):
                        self.indexes.append(idx)
                        self.range.append([i * time_step, (i + 1) * time_step])
                    if (length// time_step) * time_step < length - 4:
                        self.indexes.append(idx)
                        self.range.append([(length// time_step) * time_step, length])

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.indexes[idx]
        rg = self.range[idx]
        data = None
        #imgs = self.data[str(index)][rg[0]:rg[1]] # T x H x W x 3
        imgs = np.rollaxis(self.data[str(index)][rg[0]:rg[1]], 3, 1) # T x 3 x H x W
        for img in imgs:
            #img = self.transform(image=img)['image']
            #img = torch.as_tensor(np.rollaxis(img, 2, 0)).unsqueeze(0)
            img = self.transform(torch.as_tensor(img)).unsqueeze(0)
            data = torch.cat((data, img), dim=0) \
                if data is not None else img

        return data, self.label_dict[index], len(data), index


    def __len__(self):
        return len(self.indexes)




class test_dataset(Dataset):
    """
    images[i]: [Exxxxx/xxx.jpg, Exxxxx/xxx.jpg, ...] (T)
    """
    
    def __init__(self, data_h5, label_dict, transform, time_step=None):
        self.label_dict = label_dict
        self.data = None
        self.data_h5 = data_h5
        self.transform = transform

        self.indexes = []
        self.range = []
        with h5py.File(data_h5, 'r') as f:
            for idx in label_dict:
                length = len(f[idx])
                if time_step is None:
                    self.indexes.append(idx)
                    self.range.append([0, length])
                else:
                    for i in range(length// time_step):
                        self.indexes.append(idx)
                        self.range.append([i * time_step, (i + 1) * time_step])
                    if (length// time_step) * time_step < length - 2:
                        self.indexes.append(idx)
                        self.range.append([(length// time_step) * time_step, length])

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.indexes[idx]
        rg = self.range[idx]
        datas = []
        data = None
        imgs = np.rollaxis(self.data[str(index)][rg[0]:rg[1]], 3, 1)  # T x 3 x H x W
        for transform in self.transform:

            for img in imgs:
                img = transform(torch.as_tensor(img)).unsqueeze(0)
                data = torch.cat((data, img), dim=0) \
                        if data is not None else img
            datas.append(data)
            data = None

        return datas, self.label_dict[index], len(datas[0]), index


    def __len__(self):
        return len(self.indexes)







# class multiple_instance(Dataset):
#     """
#     images[i]: [Exxxxx/xxx.jpg, Exxxxx/xxx.jpg, ...] (T)
#     """
    
#     def __init__(self, data_h5, labels, indexes, transform):
#         self.labels = labels
#         self.data = None
#         self.data_h5 = data_h5
#         self.indexes = indexes
#         self.transform = transform

#     def __getitem__(self, idx):
        
#         if (self.data is None):
#             self.data = h5py.File(self.data_h5, 'r')
#         index = self.indexes[idx]
#         data = None
#         imgs = np.rollaxis(self.data[str(index)][()], 3, 1)  # T x 3 x H x W
#         for img in imgs:
#             data = torch.cat((data, self.transform(torch.as_tensor(img)).unsqueeze(0)), dim=0) \
#                     if data is not None else self.transform(torch.as_tensor(img)).unsqueeze(0)

#         return data, self.labels[idx], len(data), index


#     def __len__(self):
#         return len(self.labels)



def collate_fn(batch):
    batch.sort(key=lambda d: len(d[0]), reverse=True)
    images, labels, nums, indexes = zip(*batch)
    images = pad_sequence(images, batch_first=True)
    return images, torch.stack(labels), \
        torch.as_tensor(nums, dtype=torch.long), indexes


def collate_test_fn(batch):
    batch.sort(key=lambda d: len(d[0][0]), reverse=True)
    images, labels, nums, indexes = zip(*batch)
    Images = list(zip(*images))
    for i in range(len(Images)):
        Images[i] = pad_sequence(Images[i], batch_first=True)
    #Images = list(zip(*Images))
    return Images, torch.stack(labels), \
        torch.as_tensor(nums, dtype=torch.long), indexes


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


def dataloader_test(data_h5, label_dict, transform, T=4, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("shuffle", True)


    #labels, indexes = [], []

    # for (index, label) in label_dict.items():
    #     labels.append(label)
    #     indexes.append(index)


    if T == 1:
        _dataset = test_single_instance(data_h5, label_dict, transform, T)
        return DataLoader(_dataset, **kwargs)
    else:
        _dataset = test_dataset(data_h5, label_dict, transform, T)
        return DataLoader(_dataset, collate_fn=collate_test_fn, **kwargs)


def dataloader_multiple(data_h5, label_dict, transform, T=4, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("shuffle", True)


    #labels, indexes = [], []

    # for (index, label) in label_dict.items():
    #     labels.append(label)
    #     indexes.append(index)


    _dataset = multiple_instance(data_h5, label_dict, transform, T)
    if kwargs['batch_size'] == 1:
        return DataLoader(_dataset, **kwargs)
    else:
        return DataLoader(_dataset, collate_fn=collate_fn, **kwargs)
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

        img = np.rollaxis(self.data[str(index)][int(place)], 2, 0)  # H x W x 3
        img = self.transform(torch.as_tensor(img))
        return img, self.labels[index], 1, index
        


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
# if __name__ == '__main__':
#     DL = dataloader_test("data.hdf5", {'ENSG00000001630': torch.as_tensor([1, 1]), 'ENSG00000002330': torch.as_tensor([0, 1])}, utils.test_transform())
#     for idx, (feat, label, num, idx) in enumerate(DL):
#         print(len(feat))
#         print(label)
#         print(num)
#         print(idx)
#         break


# def read_fold(fold):

#     imgs = glob(os.path.join(fold, "*.jpg"))
#     key = os.path.split(fold)[-1]
#     images = []
#     for img in imgs:
#         image = Image.open(img)
#         images.append(image)
#     return images, key


    # for images, key in pr.map(read_fold, folds, workers=c, maxsize=3*c):
    #     for i in range(len(images) // T):
    #         Images.append(images[i * T: (i + 1) * T])
    #         labels.append(label_dict[key])
    #         indexes.append(key)
    #     if (i + 1) * T < len(images):
    #         Images.append(images[(i + 1) * T:])
    #         labels.append(label_dict[key])
    #         indexes.append(key)
