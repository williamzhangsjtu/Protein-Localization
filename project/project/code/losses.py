import torch.nn as nn
import torch

class DoubleLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super(DoubleLoss, self).__init__()
        #self.classfy_loss = FocalLoss()
        self.classfy_loss = nn.BCELoss()
        self.reconstruct_loss = nn.MSELoss()
        self.lambd = lambd

    def forward(self, inputs, _input, target):
        B = inputs[0].shape[0]
        return self.lambd * self.classfy_loss(inputs[0], target) + \
            (1 - self.lambd) *self.reconstruct_loss(inputs[1].view(B, -1), _input.view(B, -1))

BCELoss = nn.BCELoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.bce = nn.BCELoss(reduce='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
