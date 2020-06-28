import pandas as pd
import torch
import argparse 
import utils
import yaml, os
import model as M
import h5py
import numpy as np
import random
from glob import glob
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--disc', default='evaluate_half_disc.csv', type=str)
parser.add_argument('-v', '--pred', default='evaluate_half_pred.csv', type=str)
parser.add_argument('-p', '--path', default='output', type=str)
parser.add_argument('-m', '--model', default='kfold2/fold_2/2020-06-28_00-49-37/model_acc.th', type=str)
args = parser.parse_args()
device = torch.device('cuda:0')


def evaluate():
    obj = torch.load(args.model, lambda x, y: x)
    params = obj['param']
    config = obj['config']

    out_dir = args.path
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    logger = utils.genlogger(os.path.join(out_dir, 'log.txt'))
    out_pred = os.path.join(out_dir, args.pred)
    out_disc = os.path.join(out_dir, args.disc)

    n_class = config['n_class']
    Net = torch.load('../code/resnet_nofc.th')
    model = getattr(M, config['model'])(
        Net, n_class=n_class, **config['model_param']
    )
    model.load_param(params)
    origin_model = model

    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
    transform = utils.simple_transform()
    threshold = config['threshold']

    model = model.to(device)
    model = model.eval()
    disc, con = {}, {}

    #images = glob('test_fold/*.jpg')
    f = h5py.File('../test/test_256.h5', 'r')
    
    with torch.set_grad_enabled(False):

        for key in tqdm(f.keys()):
            input = None
            features = f[key][()]
            for feature in features:
                feature = transform(torch.as_tensor(
                    np.rollaxis(feature, 2, 0)
                )).unsqueeze(0) # 1 x C x H x W
                input = torch.cat((input, feature)) \
                    if input is not None else feature

            input = input.to(device)
            prob = model(input)[0].cpu().numpy().mean(0) # 10
            pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
            disc[key] = ";".join(np.argwhere(pred == 1).reshape(-1).astype(str))
            con[key] = ";".join(np.around(prob, decimals=4).reshape(-1).astype(str))

    # with torch.set_grad_enabled(False):
    #     features = None
    #     names = []
    #     for image in images:
    #         name = image.split('/')[-1]
    #         image = np.array(Image.open(image).resize((256,256))).astype(np.float32)
    #         feature = transform(torch.as_tensor(
    #             np.rollaxis(image, 2, 0)
    #         )).unsqueeze(0).to(device) # 1 x C x H x W
    #         prob = model(feature)[0].cpu().numpy().mean(0) # 10
    #         pred = prob >= (threshold if np.max(prob) > threshold else np.max(prob))
    #         disc[name] = ";".join(np.argwhere(pred == 1).reshape(-1).astype(str))
    #         con[name] = ";".join(np.around(prob, decimals=4).reshape(-1).astype(str))


    disc_df = pd.DataFrame(disc.items(), columns=['id', 'label']).set_index('id')
    con_df = pd.DataFrame(con.items(), columns=['id', 'pred']).set_index('id')
    disc_df.to_csv(out_disc, sep=',')
    con_df.to_csv(out_pred, sep=',')

evaluate()
