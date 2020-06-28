import torch.nn as nn
import torch

class DoubleLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super(DoubleLoss, self).__init__()
        self.classfy_loss = nn.BCELoss()
        self.reconstruct_loss = nn.MSELoss()
        self.lambd = lambd

    def forward(self, inputs, _input, target):
        B = inputs[0].shape[0]
        return self.lambd * self.classfy_loss(inputs[0], target) + \
            (1 - self.lambd) *self.reconstruct_loss(inputs[1].view(B, -1), _input.view(B, -1))

BCELoss = nn.BCELoss

class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean()
        # tp = (target * outputs).sum(0)
        # fp = ((1 - target) * outputs).sum(0)
        # fn = (target * (1 - outputs)).sum(0)

        # p = tp / (tp + fp + 1e-10)
        # r = tp / (tp + fn + 1e-10)

        # f1 = 2 * p * r / (p + r + 1e-10)
        # print(f1)
        # f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        # return 1 - f1.mean()
