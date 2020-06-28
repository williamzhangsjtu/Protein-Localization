import pandas as pd
import torch
import argparse 
import utils
import yaml, os
import model as M
import h5py
import numpy as np
from PIL import Image
import torchvision

obj_1 = torch.load('model_acc.th', lambda x, y: x)
params_1 = obj_1['param']
config = obj_1['config']
Net = torch.load('../code/resnet_nofc.th')
model_1 = getattr(M, config['model'])(
    Net, n_class=10, **config['model_param']
)
model_1.load_param(params_1)
torch.save(model_1, 'm.th')
transform = utils.simple_transform()
threshold = config['threshold']

model_1 = model_1.eval()

f = h5py.File('../code/data_256.h5', 'r')
features = f["ENSG00000173230"][()]
print(features[0])
input = None
for feature in features:
    image = transform(torch.as_tensor(
        np.rollaxis(feature, 2, 0)
    )).unsqueeze(0)
    input = torch.cat((input, image)) if input is not None else image
with torch.set_grad_enabled(False):
    output = model_1(input)[0].cpu().numpy()
    print(output)
    print(output.shape)
    prob_1 = output.mean(0)
    #prob = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
    print(prob_1)
    print(prob_1 >= (threshold if np.max(prob_1) > threshold else np.max(prob_1)))
