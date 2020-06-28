import torch, torchvision
from torch import optim
import losses
from torchvision import transforms
import pandas as pd
import numpy as np 
import yaml
import argparse, os
import time
from dataloader import dataloader_single, oversample_dataloader
import model as M
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config/baseline.yaml', type=str)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-p', '--path', default=None, type=str)
parser.add_argument('-f', '--file', default='model.th', type=str)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)



def get_config(config):
    with open(config) as c:
        data = yaml.safe_load(c)
    return data

def one_epoch(model, optimizer, criterion, dataloader, iftrain=True, grad_clip=None):
    loss_mean = 0
    if (iftrain): model = model.train()
    else: model = model.eval()

    for idx, (images, labels, nums, _,) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(iftrain):
            outputs = model(images, nums)
            loss = criterion(outputs, images, labels)
            loss_mean += loss.cpu().item()
            if (iftrain):
               optimizer.zero_grad()
               loss.backward()
               if grad_clip:
                   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
               optimizer.step() 
    return loss_mean / idx


    

def run(config_file):
    config = get_config(config_file)

    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    outdir = os.path.join(config['outputdir'], cur_time)
    os.makedirs(outdir)

    logger = utils.genlogger(os.path.join(outdir, 'log.txt'))
    logger.info("Output Path: {}".format(outdir))
    logger.info("<---- config details ---->")
    for key in config:
        logger.info("{}: {}".format(key, config[key]))
    logger.info("<---- end of config ---->")

    n_class = config['n_class']


    train_set = pd.read_csv(config['train_dev'], sep=',').values
    #test_set = pd.read_csv(config['test'], sep=',').values
    num = 10 if args.debug else None

    train_label = utils.one_hot(train_set, n_class, num)
    #test_label = utils.one_hot(test_set, n_class, num)
    logger.info("train set: {} samples".format(len(train_label)))
    #logger.info("test set: {} samples".format(len(test_label)))

    Net = torch.load(config['Net'])

    model = getattr(M, config['model'])(
        Net, n_class=n_class, **config['model_param']
    )
    if config['pretrain']:
        obj = torch.load(config['pretrain_model'], lambda x, y: x)
        model.load_param(obj['param'])
        logger.info('load from {}'.format(config['pretrain_model']))
    logger.info("model: {}".format(str(model)))
    origin_model = model

    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    logger.info("Use {} GPU(s)".format(torch.cuda.device_count()))

    model = model.to(device)

    optimizer = getattr(optim, config['optim'])(
        origin_model.parameters(), 
        lr=config['other_lr']
    )


    criterion = getattr(losses, config['Loss'])(**config['Loss_param'])

    train_transform = utils.train_transform()
    #test_transform = utils.simple_transform()

    train_dataloader = oversample_dataloader(
        config['data_h5'], train_label, train_transform, \
        T = config['time_step'], **config['dataloader_param']
    )

    # test_dataloader = dataloader_single(
    #     config['data_h5'], test_label, test_transform, \
    #     T = config['time_step'], **config['dataloader_param']
    # )


    # test_loss, f1_macro, f1_micro, acc, auc = utils.evaluate(
    #     model, test_dataloader, device, 
    #     criterion, config['threshold']
    # )
    # best_f1 = f1_macro + f1_micro
    # logger.info("test_loss: {:.4f}\tf1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}"\
    #     .format(test_loss, f1_macro, f1_micro, acc, auc))
    for epoch in range(1, config['n_epoch'] + 1):
        logger.info("<---- Epoch: {} start ---->".format(epoch))

        #if (epoch >= 10 and config['model_param']['Net_grad']): 
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr'] / 1000
        train_loss = one_epoch(
            model, optimizer, criterion,
             train_dataloader, True, config['grad_clip']
        )

        # test_loss, f1_macro, f1_micro, acc, auc = utils.evaluate(
        #     model, test_dataloader, device, 
        #     criterion, config['threshold']
        # )
        # logger.info("train_loss: {:.4f}\tdev_loss: {:.4f}".format(train_loss, test_loss))

        # logger.info("TEST: f1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}".format(f1_macro, f1_micro, acc, auc))


        if epoch % config['saveinterval'] == 0:
            model_path = os.path.join(outdir, 'model_{}.th'.format(epoch))
            torch.save({
                "param": origin_model.get_param(),
                #"train_label": train_label,
                #"test_label": test_label,
                "config": config
            }, model_path)

        model_path = os.path.join(outdir, 'model.th')
        torch.save({
            "param": origin_model.get_param(),
            #"train_label": train_label,
            "config": config
        }, model_path)
       # best_f1 = f1_macro + f1_micro
        

        # if best_f1 < f1_macro + f1_micro:
        #     model_path = os.path.join(outdir, 'model_acc.th')
        #     torch.save({
        #         "param": origin_model.get_param(),
        #         "train_label": train_label,
        #         "test_label": test_label,
        #         "config": config
        #     }, model_path)
        #     best_f1 = f1_macro + f1_micro


    # _, f1_macro, f1_micro, acc, auc = utils.evaluate(
    #     model, test_dataloader, 
    #     device, None, config['threshold']
    # )

    # logger.info("<---- test evaluation: ---->")
    # logger.info("f1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}".format(f1_macro, f1_micro, acc, auc))





if __name__ == '__main__':
    run(args.config)

