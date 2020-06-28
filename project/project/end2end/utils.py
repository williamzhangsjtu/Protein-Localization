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

class RotationTransform:
    def __init__(self, angles = [-180, -90, 0, 90, 180]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)


def train_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        RotationTransform(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def test_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
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


def evaluate_folder(model, dataloader, device, criterion=None, threshold=0.5):
    loss_mean, count = 0, 1e-10
    outputs, labels, probs = [], [], []
    with torch.set_grad_enabled(False):
        for data, label, length, _ in dataloader:
            count += 1
            data = data.to(device)
            output = model(data, length).cpu()
            if criterion:
               loss = criterion(output, label)
               loss_mean += loss.cpu().item()
            label = label.numpy()
            output = output.numpy()
            
            for i in range(len(output)):
                prob = output[i]
                pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
                outputs.append(pred.astype(int))
                labels.append(label[i])
                probs.append(prob)
    
    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    acc = accuracy_score(np.stack(labels), np.stack(outputs))
    auc = roc_auc_score(np.stack(labels), np.stack(probs), average='macro')
    return loss_mean / count, f1_macro, f1_micro, acc, auc









def evaluate(model, dataloader, device, criterion=None, threshold=0.5):
    index_len, index_dict, label_dict = {}, {}, {}
    loss_mean = 0
    model = model.eval()
    count = 0.1
    with torch.set_grad_enabled(False):
        for data, labels, indexes in dataloader:
            count += 1
            data = data.to(device)
            output = model(data)
            labels = labels.to(device)
            if criterion:
               loss = criterion(output, data, labels)
               loss_mean += loss.cpu().item()
            labels = labels.cpu().numpy()
            output = output[0].cpu().numpy()

            for i in range(len(output)):
                index_dict[indexes[i]] = index_dict.setdefault(
                    indexes[i], np.zeros(output.shape[-1])) + output[i]
                label_dict[indexes[i]] = labels[i]
                index_len[indexes[i]] = index_len.setdefault(
                    indexes[i], 0) + 1 

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

