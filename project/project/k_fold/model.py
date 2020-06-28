import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils 

class decoder(nn.Module):
    """
    input: B x D
    output: 
    """

    def __init__(self, inputdim):
        super(decoder, self).__init__()
        self.proj = nn.Linear(inputdim, 3 * 8 * 7 * 7)
        cnn_module = nn.ModuleList()
        channel = [24, 24, 12, 6, 6, 3]
        for i in range(len(channel) - 1):
            cnn_module.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            cnn_module.append(nn.BatchNorm2d(channel[i + 1]))
            cnn_module.append(nn.ReLU())
        cnn_module[-1] = nn.Tanh()
        self.cnn_module = nn.Sequential(*cnn_module)


    def forward(self, input):
        proj = self.proj(input)
        proj = proj.view(input.shape[0], 3 * 8, 7, 7)

        return self.cnn_module(proj)#.squeeze(1)


class gen_mask(nn.Module):
    """ 
    input: B x C x W x H
    output: B x D; elements in (0, 1)
    """

    def __init__(self, graph_size=256):
        super(gen_mask, self).__init__()
        down_sampling = nn.ModuleList()
        up_sampling = nn.ModuleList()

        channel = [24, 24, 12, 6, 6, 3]
        for i in range(len(channel) - 1):
            up_sampling.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            up_sampling.append(nn.BatchNorm2d(channel[i + 1]))
            up_sampling.append(nn.ReLU())
        up_sampling.append(nn.Sigmoid())

        channel = channel[::-1]
        for i in range(len(channel) - 1):
            down_sampling.append(nn.Conv2d(channel[i], channel[i + 1], 3, 2, 1))
            down_sampling.append(nn.BatchNorm2d(channel[i + 1]))
            down_sampling.append(nn.ReLU())
        
        self.up_sampling = nn.Sequential(*up_sampling)
        self.down_sampling = nn.Sequential(*down_sampling)

    def forward(self, input):
        down_sample = self.down_sampling(input)
        return self.up_sampling(down_sample)




class extractor_512(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(extractor_512, self).__init__()
        self.ResNet = ResNet
        
        self.Net_grad = Net_grad

        self.other = nn.ModuleDict({
            'decoder': decoder(512),
            'projection': nn.Linear(2048, 512),
            'OutputLayer': nn.Sequential(
                nn.Linear(512, n_class), nn.Sigmoid()
            )
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def get_Net_param(self):
        return self.ResNet.parameters()
    
    def get_other_param(self):

        return self.other.parameters()

    def get_param(self):

        return self.other.state_dict() if not self.Net_grad \
            else self.state_dict()
    
    def load_param(self, state_dict):

        if not self.Net_grad:
            self.other.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def extract_features(self, input):
        """
        input: B x C x W x H
        output: B x D
        
        """
        DenseOut = self.ResNet(input)  # B x D
        return DenseOut

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """


        DenseOut = torch.flatten(self.ResNet(input), 1)  # B x D
        DenseOut = self.other['projection'](DenseOut)

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H


class Resnet_eachlayer(nn.Module):
    

    def __init__(self, resnet, n_class, Net_grad=True):
        super(Resnet_eachlayer, self).__init__()

        self.resnet = resnet
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        
        self.Net_grad = Net_grad
        
        pretrain_dim = 512

        self.other = nn.ModuleDict({
            # 'mask_gen': gen_mask(),
            'projection': nn.Linear(3840, pretrain_dim),
            'decoder': decoder(pretrain_dim),
            'OutputLayer': nn.Sequential(
                nn.Linear(pretrain_dim, n_class), nn.Sigmoid()
            )
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.normal_(self.other['projection'].weight, 0, 0.1)
        nn.init.constant_(self.other['projection'].bias, 0)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def get_Net_param(self):
        return self.ResNet.parameters()
    
    def get_other_param(self):

        return self.other.parameters()

    def get_param(self):

        return self.other.state_dict() if not self.Net_grad \
            else self.state_dict()
    
    def load_param(self, state_dict):

        if not self.Net_grad:
            self.other.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """
        # mask = self.other['mask_gen'](input)
        # mask = mask.to(torch.float)
        # input = mask * input


        layer_1 = self.resnet[:5](input)
        layer_2 = self.resnet[5:6](layer_1)
        layer_3 = self.resnet[6:7](layer_2)
        layer_4 = self.resnet[7:](layer_3)
        
        layer_1 = self.pooling(layer_1).flatten(1)
        layer_2 = self.pooling(layer_2).flatten(1)
        layer_3 = self.pooling(layer_3).flatten(1)
        layer_4 = layer_4.flatten(1)

        output = torch.cat((
            layer_1, layer_2,
            layer_3, layer_4
        ), dim=1)

        output = self.other['projection'](output)
        
        

        return self.other['OutputLayer'](output), \
            self.other['decoder'](output) # B x C x W x H
