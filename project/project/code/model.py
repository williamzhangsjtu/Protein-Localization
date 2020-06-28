import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils 



class LayerOut:
    
    def __init__(self, modules):
        self.Pooling = nn.AdaptiveAvgPool2d(1)
        self.features = {}

        self.hooks = [
            module.register_forward_hook(self.hook_fn)
            for module in modules
        ]

    def hook_fn(self, module, input, output):
        self.features[module] = self.Pooling(output).view(output.shape[0], -1)

    def remove(self):

        for hook in self.hooks:
            hook.remove()





class AdapativeAttention(nn.Module):
    """
    input[i]: B x D 
    """

    def __init__(self, *dim, com_dim=1024):
        super(AdapativeAttention, self).__init__()
        self.module = nn.ModuleList([
            nn.Linear(d, com_dim) for d in dim
        ])

        self.SimpleAttn = SimpleAttn(com_dim)
    def forward(self, input):
        projs = self.module[0](input[0]).unsqueeze(1)
        for i in range(1, len(self.module)):
            projs = torch.cat((projs, self.module[i](input[i]).unsqueeze(1)),dim=1)

        return self.SimpleAttn(projs)



class SimpleAttn(nn.Module):
    def __init__(self, dim):
        super(SimpleAttn, self).__init__()
        self.Linear = nn.Linear(dim, 1)
        self.Softmax = nn.Softmax(dim=1)
        nn.init.normal_(self.Linear.weight, 0, 0.1)
        nn.init.constant_(self.Linear.bias, 0.0)

    def forward(self, input, input_num=None):
        if (input_num is not None):
            idxs = torch.arange(input.shape[1]).repeat(input.shape[0]).view(input.shape[:2])
            masks = idxs.cpu() < input_num.cpu().view(-1, 1)
            masks = masks.to(torch.float).to(input.device)
            input = input * masks.unsqueeze(-1)
        
        alpha = self.Softmax(self.Linear(input))
        output = (alpha * input).sum(1)
        return output # B x D



class TransformerBlock1(nn.Module):

    def __init__(self, DenseNet, n_class=10, n_head=6, n_layer=4, n_dim=2048):
        super(TransformerBlock1, self).__init__()
        self.DenseNet = DenseNet.features[:5]   # first block  6 is the second 8 is the third
        self.Pooling = nn.AdaptiveAvgPool2d(1)
        
        pretrain_dim = self.__getDenseDim()
        encoder_layer = nn.TransformerEncoderLayer(pretrain_dim, n_head, dim_feedforward=n_dim)

        
        self.Transformer = nn.ModuleDict({
            'Encoder': nn.TransformerEncoder(encoder_layer, n_layer),
            'Attention': SimpleAttn(pretrain_dim),
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class))
        })

        nn.init.normal_(self.Transformer['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.Transformer['OutputLayer'][0].bias, 0)
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            dim = self.DenseNet(test).shape[1]
        return dim


    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_Transformer_param(self):

        return self.Transformer.parameters()
    
    def get_param(self):
        return self.state_dict()

    def load_param(self, state_dict):
        self.load_state_dict(state_dict)


    def forward(self, input, input_num=None):
        """
        input: B x T x C x W x H
        input_num: B
        """
        B, T = input.shape[:2]

        DenseOut = self.DenseNet(input.view(-1, *input.shape[2:]))
        EncoderIn = self.Pooling(DenseOut).view(B, T, -1).transpose(0, 1)

        if B == 1: input_num = None

        if (input_num is not None):
            idxs = torch.arange(T).unsqueeze(0).repeat(B, 1)
            masks = idxs.cpu() >= input_num.unsqueeze(-1).cpu()
            masks = masks.bool().to(input.device)
            tsf_out = self.Transformer['Encoder'](
                EncoderIn, src_key_padding_mask=masks).transpose(0, 1)
        else:
            tsf_out = self.Transformer['Encoder'](EncoderIn).transpose(0, 1)

        attn = self.Transformer['Attention'](tsf_out, input_num) # B x D
        output = self.Transformer['OutputLayer'](attn)

        return output





class TransformerModel(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True, n_layer=1, **kwargs):
        super(TransformerModel, self).__init__()
        self.DenseNet = DenseNet.features[:7]
        self.denseblocks = [
            self.DenseNet.denseblock1,
            #self.DenseNet.denseblock2,
            #self.DenseNet.denseblock3,
            #self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad

        pretrain_dim = self.__getDenseDim()
        encoder_layer = nn.TransformerEncoderLayer(pretrain_dim, dim_feedforward=2*pretrain_dim, **kwargs)
        encoder_layer.self_attn.kdim = 32
        encoder_layer.self_attn.vdim = 32
        
        self.other = nn.ModuleDict({
            #'Projection': nn.Linear(pretrain_dim, n_dim),
            'Encoder': nn.TransformerEncoder(encoder_layer, n_layer),
            'Attention': SimpleAttn(pretrain_dim),
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
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
        input: B x T x C x W x H
        input_num: B
        """

        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:])).view(B, T, -1).transpose(0, 1)

        if (input_num is not None):
            idxs = torch.arange(T).unsqueeze(0).repeat(B, 1)
            masks = idxs.cpu() >= input_num.unsqueeze(-1).cpu()
            masks = masks.bool().to(input.device)
            fusion = self.other['Encoder'](
                DenseOut, src_key_padding_mask=masks).transpose(0, 1)
        else:
            fusion = self.other['Encoder'](DenseOut).transpose(0, 1)

        attn = self.other['Attention'](fusion, input_num)
        return self.other['OutputLayer'](attn)



class MultiheadFusion(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True, nhead=1, n_dim=1024):
        super(MultiheadFusion, self).__init__()
        self.DenseNet = DenseNet.features#[:9]
        self.denseblocks = [
            self.DenseNet.denseblock1,
            self.DenseNet.denseblock2,
            self.DenseNet.denseblock3,
            self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad

        pretrain_dim = self.__getDenseDim()
        n_dim = 3 * 8 * 8 * 8
        
        self.other = nn.ModuleDict({
            'Projection': nn.Linear(pretrain_dim, n_dim),
            'Fusion': nn.MultiheadAttention(n_dim, nhead),
            'Attention': SimpleAttn(n_dim),
            'OutputLayer': nn.Sequential(nn.Linear(n_dim, n_class), nn.Sigmoid()),
            'reconstructor': decoder(n_dim)
        })
        #nn.init.normal_(self.other['Projection'].weight, 0, 0.1)
        #nn.init.constant_(self.other['Projection'].bias, 0)
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
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
        input: B x T x C x W x H
        input_num: B
        """

        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:]))
        dense_proj = self.other['Projection'](DenseOut)
        projection = dense_proj.view(B, T, -1).transpose(0, 1)

        #projection = DenseOut.view(B, T, -1).transpose(0, 1)
        if (input_num is not None):
            idxs = torch.arange(T).unsqueeze(0).repeat(B, 1)
            masks = idxs.cpu() >= input_num.unsqueeze(-1).cpu()
            masks = masks.bool().to(input.device)
            fusion = self.other['Fusion'](
                projection, projection, projection, key_padding_mask=masks)[0].transpose(0, 1)
        else:
            fusion = self.other['Fusion'](projection)[0].transpose(0, 1)

        attn = self.other['Attention'](fusion, input_num)
        
        return self.other['OutputLayer'](attn), \
                self.other['reconstructor'](dense_proj).view(*input.shape)



class ConvolutionFusion(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(ConvolutionFusion, self).__init__()
        self.DenseNet = DenseNet.features
        self.denseblocks = [
            self.DenseNet.denseblock1,
            self.DenseNet.denseblock2,
            self.DenseNet.denseblock3,
            self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad

        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(4)
        )
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 5), padding=1, stride=(2, 3)),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(16)
        )
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 5), padding=1, stride=(2, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        
        self.other = nn.ModuleDict({
            'cnn': nn.Sequential(
                cnn_block1, cnn_block2, cnn_block3, 
                nn.AdaptiveAvgPool2d(1)
            ),
            'OutputLayer': nn.Sequential(nn.Linear(256, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
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
        input: B x T x C x W x H
        input_num: B
        """

        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:])).view(B, T, -1)

        cnn_out = self.other['cnn'](DenseOut.unsqueeze(1)).view(B, -1)
        return self.other['OutputLayer'](cnn_out)




class Baseline_Variance(nn.Module):

    def __init__(self, DenseNet, n_class, com_dim=1024):
        super(Baseline_Variance, self).__init__()
        self.DenseNet = DenseNet
        self.denseblocks = [
            self.DenseNet.features.denseblock1,
            self.DenseNet.features.denseblock2,
            self.DenseNet.features.denseblock3,
            self.DenseNet.features.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        
        
        pretrain_dim = self.__getDenseDim()


        self.OtherPart = nn.Sequential(
            AdapativeAttention(*pretrain_dim),
            nn.Linear(com_dim, n_class), 
            nn.Sigmoid()
        )
        nn.init.normal_(self.OtherPart[1].weight, 0, 0.1)
        nn.init.constant_(self.OtherPart[1].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)

        dim = [self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks]
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)

        self.DenseNet(input)

        output = list(self.BlockOut.features.values())
        return output  

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_other_param(self):

        return self.OtherPart.parameters()

    def get_param(self):

        return self.OtherPart.state_dict()
    
    def load_param(self, state_dict):

        self.OtherPart.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """


        DenseOut = self._getDenseOut(input)


        return self.OtherPart(DenseOut)





class Baseline(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(Baseline, self).__init__()
        self.DenseNet = DenseNet#.features[:9]
        self.denseblocks = [
            self.DenseNet.features.denseblock1,
            self.DenseNet.features.denseblock2,
            self.DenseNet.features.denseblock3,
            self.DenseNet.features.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)

        
        pretrain_dim = self.__getDenseDim()
        self.other = nn.ModuleDict({
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)

        self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_other_param(self):

        return self.other.parameters()

    def get_param(self):

        return self.state_dict()
    
    def load_param(self, state_dict):

        self.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """


        DenseOut = self._getDenseOut(input)
        
        

        return self.other['OutputLayer'](DenseOut)



class Baseline_ResNet(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(Baseline_ResNet, self).__init__()
        self.ResNet = ResNet
        self.reslayer = [
            self.ResNet.layer1,
            self.ResNet.layer2,
            self.ResNet.layer3,
            self.ResNet.layer4,
        ]
        self.BlockOut = LayerOut(self.reslayer)
        self.Net_grad = Net_grad
        
        pretrain_dim = self.__getResDim()
        self.other = nn.ModuleDict({
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getResDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.ResNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.reslayer)
        return dim

    def _getResOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.ResNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.ResNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D


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


        DenseOut = self._getResOut(input)
        
        

        return self.other['OutputLayer'](DenseOut)



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



class Reconstructor_ResNet(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(Reconstructor_ResNet, self).__init__()
        self.ResNet = ResNet
        self.reslayer = [
            self.ResNet.layer1,
            self.ResNet.layer2,
            self.ResNet.layer3,
            self.ResNet.layer4,
        ]
        self.BlockOut = LayerOut(self.reslayer)
        self.Net_grad = Net_grad
        
        pretrain_dim = self.__getResDim()
        self.other = nn.ModuleDict({
            'decoder': decoder(pretrain_dim),
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getResDim(self):
        test = torch.zeros(1, 3, 512, 512)
        with torch.set_grad_enabled(False):
            self.ResNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.reslayer)
        return dim

    def _getResOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.ResNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.ResNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D


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


        DenseOut = self._getResOut(input)  # B x D
        
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H


class Baseline_decoder(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(Baseline_decoder, self).__init__()
        self.DenseNet = DenseNet#.features[:9]
        self.denseblocks = [
            self.DenseNet.features.denseblock1,
            self.DenseNet.features.denseblock2,
            self.DenseNet.features.denseblock3,
            self.DenseNet.features.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad
        
        pretrain_dim = self.__getDenseDim()
        self.OtherPart = nn.ModuleDict({
            'classifier': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid()), 
            'reconstructor': decoder(pretrain_dim)
        })
        nn.init.normal_(self.OtherPart['classifier'][0].weight, 0, 0.1)
        nn.init.constant_(self.OtherPart['classifier'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D


    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_other_param(self):

        return self.OtherPart.parameters()

    def get_param(self):

        return self.OtherPart.state_dict() if not self.Net_grad \
            else self.state_dict()
    
    def load_param(self, state_dict):

        if not self.Net_grad:
            self.OtherPart.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """
        DenseOut = self._getDenseOut(input)

        return self.OtherPart['classifier'](DenseOut), \
            self.OtherPart['reconstructor'](DenseOut)




class FusionReconstructor(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(FusionReconstructor, self).__init__()
        self.DenseNet = DenseNet.features
        self.denseblocks = [
            self.DenseNet.denseblock1,
            self.DenseNet.denseblock2,
            self.DenseNet.denseblock3,
            self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad
        pretrain_dim = self.__getDenseDim()

        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(4)
        )
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 5), padding=1, stride=(1, 3)),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(16)
        )
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 5), padding=1, stride=(2, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        
        self.other = nn.ModuleDict({
            'cnn': nn.Sequential(
                cnn_block1, cnn_block2, cnn_block3,
                nn.AdaptiveAvgPool2d(1)
            ),
            'projection': nn.Linear(pretrain_dim, 3 * 8 * 8 * 8),
            'reconstructor': decoder(3 * 8 * 8 * 8), 
            'OutputLayer': nn.Sequential(nn.Linear(64, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
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
        input: B x T x C x W x H
        input_num: B
        """

        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:]))
        DenseOut = self.other['projection'](DenseOut)

        cnn_out = self.other['cnn'](
            DenseOut.view(B, T, -1).unsqueeze(1)).view(B, -1)
        return self.other['OutputLayer'](cnn_out), \
            self.other['reconstructor'](DenseOut).view(*input.shape)  # B x T x C x W x H
# if __name__ == "__main__":
#     a = AdapativeAttention(3,4,5)




class FusionReconstructor_ResNet(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(FusionReconstructor_ResNet, self).__init__()
        self.ResNet = ResNet
        self.reslayer = [
            self.ResNet.layer1,
            self.ResNet.layer2,
            self.ResNet.layer3,
            self.ResNet.layer4,
        ]
        self.BlockOut = LayerOut(self.reslayer)
        self.Net_grad = Net_grad
        pretrain_dim = self.__getDenseDim()

        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(4)
        )
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 5), padding=1, stride=(1, 3)),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(16)
        )
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 5), padding=1, stride=(2, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        
        self.other = nn.ModuleDict({
            'cnn': nn.Sequential(
                cnn_block1, cnn_block2, cnn_block3,
                nn.AdaptiveAvgPool2d(1)
            ),
            'reconstructor': decoder(pretrain_dim), 
            'OutputLayer': nn.Sequential(nn.Linear(64, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.ResNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.reslayer)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.ResNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.ResNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

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
        input: B x T x C x W x H
        input_num: B
        """

        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:]))

        cnn_out = self.other['cnn'](
            DenseOut.view(B, T, -1).unsqueeze(1)).view(B, -1) # B x D
        return self.other['OutputLayer'](cnn_out), \
            self.other['reconstructor'](DenseOut).view(*input.shape)  # (B x T) x C x W x H











class UNet(nn.Module):
    """
    input: B x T x 3 x 256 x 256
    """

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(UNet, self).__init__()

        self.downsampling = nn.ModuleDict(
            {'block1': nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(8), # 8 x 128 x 128
            ),

            'block2': nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(16), # 16 x 64 x 64
            ),

            'block3': nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(32), # 32 x 32 x 32
            ),

            'block4': nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(64), # 64 x 16 x 16
            ),

            'block5': nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(128), # 128 x 8 x 8
            ),

            'block6': nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2), 
                nn.ReLU(), nn.BatchNorm2d(256), # 256 x 4 x 4
            )
        })

            
        self.upsampling = nn.ModuleDict({
            'block1': nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
            ),

            'block2': nn.Sequential(
                nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
            ),

            'block3': nn.Sequential(
                nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(),
            ),

            'block4': nn.Sequential(
                nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(),
            ),

            'block5': nn.Sequential(
                nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8), nn.ReLU(),
            ),

            'block6': nn.Sequential(
                nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(3), nn.ReLU(),
            )
        })

        self.Pooling = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Sequential(
            nn.Linear(256, n_class),
            nn.Sigmoid()
        )

    def get_param(self):

        return self.state_dict()
    
    def load_param(self, state_dict):

        self.load_state_dict(state_dict)


    def forward(self, input, nums=None):
        B = input.shape[0]
        down_block1 = self.downsampling['block1'](input) # 8 x 128 x 128
        down_block2 = self.downsampling['block2'](down_block1) # 16 x 64 x 64
        down_block3 = self.downsampling['block3'](down_block2) # 32 x 32 x 32
        down_block4 = self.downsampling['block4'](down_block3) # 64 x 16 x 16
        down_block5 = self.downsampling['block5'](down_block4) # 128 x 8 x 8
        down_block6 = self.downsampling['block6'](down_block5) # 128 x 8 x 8

        prob = self.output(self.Pooling(down_block6).view(B, -1))

        up_block1 = self.upsampling['block1'](down_block6)
        up_block2 = self.upsampling['block2'](torch.cat((down_block5, up_block1), dim=1))
        up_block3 = self.upsampling['block3'](torch.cat((down_block4, up_block2), dim=1))
        up_block4 = self.upsampling['block4'](torch.cat((down_block3, up_block3), dim=1))
        up_block5 = self.upsampling['block5'](torch.cat((down_block2, up_block4), dim=1))
        up_block6 = self.upsampling['block6'](torch.cat((down_block1, up_block5), dim=1))

        return prob, up_block6








class GapNet(nn.Module):

    def __init__(self, DenseNet, n_class, Net_grad=True):
        super(GapNet, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.Net_grad = Net_grad
        cnn_block1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        cnn_block2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        cnn_block3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU()
        )

        self.encoder = nn.ModuleDict({
            'cnn_block1': cnn_block1,
            'cnn_block2': cnn_block2,
            'cnn_block3': cnn_block3
        })

        cat_dim = self.get_cat_dim()

        output = nn.Sequential(
            nn.Linear(cat_dim, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.Linear(256, n_class),
            nn.Sigmoid()
        )

        

        self.other = nn.ModuleDict({
            'decoder': decoder(cat_dim),
            'output': output
        })

    def get_cat_dim(self):
        input = torch.randn(1, 3, 256, 256)
        block1_out = self.encoder['cnn_block1'](input)
        block2_out = self.encoder['cnn_block2'](block1_out)
        block3_out = self.encoder['cnn_block3'](block2_out)

        return block1_out.shape[1] + block2_out.shape[1] + block3_out.shape[1]

    def forward(self, x, x_num=None):
        x = self.encoder['cnn_block1'](x)
        gap1 = self.dropout(x).mean([2, 3])

        x = self.encoder['cnn_block2'](x)
        gap2 = self.dropout(x).mean([2, 3])

        x = self.encoder['cnn_block3'](x)
        gap3 = self.dropout(x).mean([2, 3])

        x = torch.cat((gap1, gap2, gap3), dim=1)

        classify = self.other['output'](x)
        reconstruct = self.other['decoder'](x)

        return classify, reconstruct

    def get_Net_param(self):
        return self.encoder.parameters()
    
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


class Mask_ResNet(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(Mask_ResNet, self).__init__()
        self.ResNet = ResNet
        self.reslayer = [
            self.ResNet.layer1,
            self.ResNet.layer2,
            self.ResNet.layer3,
            self.ResNet.layer4,
        ]
        #self.reslayer = [
        #    self.ResNet.features.denseblock1,
        #    self.ResNet.features.denseblock2,
        #    self.ResNet.features.denseblock3,
        #    self.ResNet.features.denseblock4,
        #]
        self.BlockOut = LayerOut(self.reslayer)
        self.Net_grad = Net_grad
        pretrain_dim = self.__getDenseDim()

        self.other = nn.ModuleDict({
            'mask_gen': gen_mask(),
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 256, 256)
        with torch.set_grad_enabled(False):
            self.ResNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.reslayer)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.ResNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.ResNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

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
        input: B x T x C x W x H
        input_num: B
        """

        mask = self.other['mask_gen'](input)
        mask = (mask >= 0.5).to(torch.float)
        mask_input = mask * input
        DenseOut = self._getDenseOut(mask_input)

        return self.other['OutputLayer'](DenseOut)




class mrcnn_densenet(nn.Module):
    

    def __init__(self, DenseNet, mrcnn, n_class, Net_grad=True):
        super(mrcnn_densenet, self).__init__()
        self.DenseNet = DenseNet.features

        self.Net_grad = Net_grad

        test = torch.randn(1,3,256,256)
        
        pretrain_dim = self.DenseNet(test).shape[1]

        
        
        self.other = nn.ModuleDict({
            'pooling': torch.nn.AdaptiveAvgPool2d(1),
            'mrcnn': mrcnn,
            'OutputLayer': nn.Sequential(nn.Linear(pretrain_dim, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
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

        self.other['mrcnn'].eval()
        masks = self.other['mrcnn'](input)[0]['masks']
        if masks.shape[0] > 0:
            mask_input = (masks[0] >= 0.5).to(torch.float32) * input
        else:
            mask_input = input
        DenseOut = self.DenseNet(mask_input)
        DenseOut = self.other['pooling'](DenseOut).view(input.shape[0], -1)

        return self.other['OutputLayer'](DenseOut)


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

class Densenet_decoder(nn.Module):
    

    def __init__(self, Densenet, n_class, Net_grad=True):
        super(Densenet_decoder, self).__init__()

        self.Densenet = nn.Sequential(
            Densenet.features,
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.Net_grad = Net_grad
        

        pretrain_dim = 1920

        self.other = nn.ModuleDict({
            'decoder': decoder(pretrain_dim),
            'OutputLayer': nn.Sequential(
                nn.Linear(pretrain_dim, n_class), nn.Sigmoid()
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


        DenseOut = self.Densenet(input).flatten(1)  # B x D
        
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H

class Resnet_eachlayer(nn.Module):
    

    def __init__(self, resnet, n_class, Net_grad=True):
        super(Resnet_eachlayer, self).__init__()

        self.resnet = resnet
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        
        self.Net_grad = Net_grad
        
        pretrain_dim = 3840

        self.other = nn.ModuleDict({
            'mask_gen': gen_mask(),
            #'projection': nn.Linear(3840, pretrain_dim),
            'decoder': decoder(pretrain_dim),
            'OutputLayer': nn.Sequential(
                nn.Linear(pretrain_dim, n_class), nn.Sigmoid()
            )
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        #nn.init.normal_(self.other['projection'].weight, 0, 0.1)
        #nn.init.constant_(self.other['projection'].bias, 0)
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
        mask = self.other['mask_gen'](input)
        mask = (mask >= 0.5).to(torch.float)
        input = mask * input


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

        #output = self.other['projection'](output)
        
        

        return self.other['OutputLayer'](output), \
            self.other['decoder'](output), input # B x C x W x H
