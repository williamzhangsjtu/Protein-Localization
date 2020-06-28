from torchvision import transforms
import torchvision.transforms.functional as F
import logging, os, sys
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import random, cv2
import torch
from numpy.random import standard_normal
import numpy as np
from PIL import Image, ImageFilter
# from albumentations import (
#    Compose,
#    Resize,
#    OneOf,
#    RandomBrightness,
#    RandomContrast,
#    MotionBlur,
#    MedianBlur,
#    GaussianBlur,
#    VerticalFlip,
#    HorizontalFlip,
#    ShiftScaleRotate,
#    Normalize,
# )


# def augmentation():
#     return Compose([
#        #Resize(height=224, width=224),
#        #OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
#        #OneOf([MotionBlur(blur_limit=3),MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3),], p=0.5,),
#        VerticalFlip(p=0.5),
#        HorizontalFlip(p=0.5),
#        ShiftScaleRotate(
#             shift_limit=0.2,
#             scale_limit=0.2,
#             rotate_limit=20,
#             interpolation=cv2.INTER_LINEAR,
#             border_mode=cv2.BORDER_REFLECT_101,
#             p=1,
#        ),
#        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
#     ], p=1.0)

class RotationTransform:
    def __init__(self, angles = [-180, -90, 0, 90, 180]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)


class RandomMask:
    def __init__(self, p=0.1):
        self.p = p
        self.var_limit = [0.0, 5.0]
        self.mean = 0

    def __call__(self, img):
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        gauss = torch.normal(self.mean, var ** 2, (3, 224, 224))
        mask = torch.randint(0, 10, (3, 224, 224)) >= 1
        return mask.to(float) * gauss.to(float) + img.to(float)


# class GaussianBlur:
#     def __init__(self, radius = 2):
#         self.radius = radius

#     def __call__(self, img):
#         img2 = img.filter(ImageFilter.GaussianBlur(radius = self.radius)) 
#         return img2

# class GaussianNoise:
#     def __init__(self, snr=0.1, mean=0):
#         self.mean = mean
#         self.snr = snr
        
#     def __call__(self, img):
#         vs = np.var(img)
#         vn = vs / self.snr
#         sd = np.sqrt(vn)
#         s = np.ndarray(img.shape)
#         noise = sd * standard_normal(s.shape) + self.mean
#         t = img + torch.from_numpy(noise)

#         return t


# def augmentation_transform():
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomHorizontalFlip(), 
#         transforms.RandomVerticalFlip(),
#         RotationTransform(),
#         transforms.RandomCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform



# def augmentation_transform():
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         GaussianBlur(),
#         transforms.RandomHorizontalFlip(), 
#         transforms.RandomVerticalFlip(),
#         RotationTransform(),
#         transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
#         transforms.RandomCrop(224),
#         transforms.ToTensor(),
#         GaussianNoise(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         GaussianNoise(),
#     ])
#     return transform
def train_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        RotationTransform(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def test_transform():
    transform = [
        
        transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(224),
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(224),
            RotationTransform(),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    ]
    return transform

def simple_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def genlogger(outputfile):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(logging.INFO)
    stdlog = logging.StreamHandler(sys.stdout)
    stdlog.setFormatter(formatter)
    file_handler = logging.FileHandler(outputfile)
    file_handler.setFormatter(formatter)
    # Log to stdout
    logger.addHandler(file_handler)
    logger.addHandler(stdlog)
    return logger



def train_dev_test_split(df, outdir):
    length = len(df)
    shuffle = np.arange(length)
    random.shuffle(shuffle)
    train_set = df.values[shuffle[:int(length * 0.7)]]
    dev_set = df.values[shuffle[int(length * 0.7):int(length * 0.85)]]
    test_set = df.values[shuffle[int(length * 0.85):]]

    train_df = pd.DataFrame(train_set, columns=['id', 'label'])
    train_df.set_index('id').to_csv(os.path.join(outdir, 'train.csv'))
    dev_df = pd.DataFrame(dev_set, columns=['id', 'label'])
    dev_df.set_index('id').to_csv(os.path.join(outdir, 'dev.csv'))
    test_df = pd.DataFrame(test_set, columns=['id', 'label'])
    test_df.set_index('id').to_csv(os.path.join(outdir, 'test.csv'))

    return train_set, dev_set, test_set

def train_dev_split(df, outdir):
    length = len(df)
    shuffle = np.arange(length)
    random.shuffle(shuffle)
    train_set = df.values[shuffle[:int(length * 0.82)]]
    dev_set = df.values[shuffle[int(length * 0.82):]]


    train_df = pd.DataFrame(train_set, columns=['id', 'label'])
    train_df.set_index('id').to_csv(os.path.join(outdir, 'train.csv'))
    dev_df = pd.DataFrame(dev_set, columns=['id', 'label'])
    dev_df.set_index('id').to_csv(os.path.join(outdir, 'dev.csv'))

    return train_set, dev_set


def one_hot(data_set, n_class, num=None):
    Dict = {idx: list(map(int, label.split(';')))  for idx, label in data_set}
    label = {}
    for idx, (key, values) in enumerate(Dict.items()):
        if (num is not None and idx > num): break
        label[key] = torch.zeros(n_class, dtype=torch.float32)
        label[key][values] = 1
    return label


def evaluate_sample(model, dataloader, device, threshold=0.5):

    outputs, labels = [], []
    with torch.set_grad_enabled(False):
        for idx, (imgs, label, nums, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            label = label.numpy()
            output = model(imgs, nums)[0].cpu().numpy()
            
            for i in range(len(output)):
                prob = output[i] - threshold
                pred = prob >= (0 if np.max(prob) > 0 else np.max(prob))
                outputs.append(pred.astype(int))
                labels.append(label[i])
    
    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    return f1_macro, f1_micro



def evaluate(model, dataloader, device, criterion=None, threshold=0.5):
    index_dict, label_dict, index_len = {}, {}, {}
    loss_mean = 0
    model = model.eval()
    count = 0.1
    with torch.set_grad_enabled(False):
        for i in range(1):

            for img, labels, nums, indexes in dataloader:

                img = img.to(device)
                output = model(img, nums)
                if (output.__class__ == tuple):
                    output = output[0].cpu().numpy()
                else:
                    output = output.cpu().numpy()

                nums = nums.numpy()
                labels = labels.numpy()

                for i in range(len(output)):
                    index_dict[indexes[i]] = index_dict.setdefault(
                        indexes[i], np.zeros(output.shape[-1])) + \
                        output[i] * nums[i]
                    label_dict[indexes[i]] = labels[i]
                    index_len[indexes[i]] = index_len.setdefault(
                        indexes[i], 0) + nums[i]
            #dataloader.dataset.re_sample()
    outputs, labels, probs = [], [], []
    for (index, prob) in index_dict.items():
        prob /= index_len[index]
        pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
        outputs.append(pred.astype(int))
        labels.append(label_dict[index])
        probs.append(prob)

    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    acc = accuracy_score(np.stack(labels), np.stack(outputs))
    try:
        auc = roc_auc_score(np.stack(labels), np.stack(probs), average='macro')
    except:
        auc = 0
    return loss_mean / count, f1_macro, f1_micro, acc, auc





def evaluate_tta(model, dataloader, device, criterion=None, threshold=0.5):
    index_dict, label_dict, index_len = {}, {}, {}
    loss_mean = 0
    model = model.eval()
    count = 0.1
    with torch.set_grad_enabled(False):
        for imgs, labels, nums, indexes in dataloader:
            count += 1
            outputs = []

            for img in imgs:
                img = img.to(device)
                output = model(img, nums)
                if (output.__class__ == tuple):
                    outputs.append(output[0].cpu().numpy())
                else:
                    outputs.append(output.cpu().numpy())
            output = sum(outputs) / len(outputs)
            #if criterion:
            #    loss = criterion(output, imgs[0].cpu(), labels)
            #    loss_mean += loss.cpu().item()
            nums = nums.numpy()
            labels = labels.numpy()

            for i in range(len(output)):
                index_dict[indexes[i]] = index_dict.setdefault(
                    indexes[i], np.zeros(output.shape[-1])) + \
                    output[i] * nums[i]
                label_dict[indexes[i]] = labels[i]
                index_len[indexes[i]] = index_len.setdefault(
                    indexes[i], 0) + nums[i]
    outputs, labels, probs = [], [], []
    for (index, prob) in index_dict.items():
        prob /= index_len[index]
        pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
        outputs.append(pred.astype(int))
        labels.append(label_dict[index])
        probs.append(prob)

    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    acc = accuracy_score(np.stack(labels), np.stack(outputs))
    try:
        auc = roc_auc_score(np.stack(labels), np.stack(probs), average='macro')
    except:
        auc = 0
    return loss_mean / count, f1_macro, f1_micro, acc, auc



def evaluate_aug(model, dataloader, device, criterion=None, threshold=0.5):
    index_dict, label_dict = {}, {}
    loss_mean = 0
    model = model.eval()
    count = 0.1
    index_len, index_dict, label_dict = {}, {}, {}
    with torch.set_grad_enabled(False):
        for idx, (imgs, noise_img, labels, nums, indexes) in enumerate(dataloader):
            count += 1
            noise_img = noise_img.to(device)
            labels = labels.numpy()
            output = model(noise_img, nums)
            if criterion:
                loss = criterion(output, imgs, labels)
                loss_mean += loss.cpu().item()
            nums = nums.numpy()
            if (output.__class__ == tuple):
                output = output[0].cpu().numpy()
            else:
                output = output.cpu().numpy()

            for i in range(len(output)):
                index_dict[indexes[i]] = index_dict.setdefault(
                    indexes[i], np.zeros(output.shape[-1])) + \
                    (output[i] - threshold) * nums[i]
                label_dict[indexes[i]] = labels[i]
                index_len[indexes[i]] = index_len.setdefault(
                    indexes[i], 0) + nums[i]
    outputs, labels, probs = [], [], []
    for (index, prob) in index_dict.items():
        prob /= index_len[index]
        pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
        outputs.append(pred.astype(int))
        labels.append(label_dict[index])
        probs.append(prob)

    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    acc = accuracy_score(np.stack(labels), np.stack(outputs))
    auc = roc_auc_score(np.stack(labels), np.stack(probs), average='macro')
    return loss_mean / count, f1_macro, f1_micro, acc, auc
