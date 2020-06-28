import torch
import os
import h5py
import argparse
import pandas as pd
import torchvision
import model as M
import utils
from tqdm import tqdm
import numpy as np
from dataloader import pretrain_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str)
parser.add_argument('-o', '--output', type=str)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obj = torch.load(args.model, lambda x, y: x)

config = obj['config']
param = obj['param']
Net = torchvision.models.resnet152(pretrained=False)

n_class = config['n_class']
model = getattr(M, config['model'])(
    Net, n_class=n_class, **config['model_param']
)

model = model.to(device)
model.load_param(param)

transform = utils.test_transform()

data_h5 = config['data_h5']
dim = model.feature_dim
with torch.set_grad_enabled(False):
    with h5py.File(os.path.join(args.output, 'feature_{}.hdf5'.format(dim)), 'w') as out:
        data = h5py.File(data_h5, 'r')
        for name in tqdm(data.keys()):
            img_np = np.rollaxis(data[str(name)][()], 3, 1)
            img_np = torch.as_tensor(img_np)
            imgs = None
            for img in img_np:
                img = transform(img).unsqueeze(0)
                imgs = torch.cat((imgs, img), dim=0) \
                        if imgs is not None else img
            outputs = None
            imgs = imgs.to(device)
            for i in range(len(imgs) // 32):
                img = imgs[i * 32: (i + 1) * 32]
                output = model.extract_features(img).cpu()
                outputs = torch.cat((output, outputs), dim=0) \
                    if outputs is not None else output
            if (len(imgs) // 32) * 32 < len(imgs):
                img = imgs[(len(imgs) // 32) * 32:]
                output = model.extract_features(img).cpu()
                outputs = torch.cat((output, outputs), dim=0) \
                    if outputs is not None else output
            print(outputs.shape)
            print(name)
            # for img in imgs:
            #     output = model.extract_features(img.unsqueeze(0)).cpu().numpy()
            #     outputs = torch.cat((output, outputs), dim=0) \
            #         if outputs is not None else output
            out[name] = outputs.numpy()


