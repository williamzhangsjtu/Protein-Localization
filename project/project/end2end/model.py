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




class decoder(nn.Module):
    """
    input: B x D
    output: 
    """

    def __init__(self, inputdim):
        super(decoder, self).__init__()
        self.proj = nn.Linear(inputdim, 3 * 4 * 8 * 8)
        cnn_module = nn.ModuleList()
        channel = [12, 12, 6, 6, 3, 3]
        for i in range(len(channel) - 1):
            cnn_module.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            cnn_module.append(nn.BatchNorm2d(channel[i + 1]))
            cnn_module.append(nn.ReLU())
        cnn_module[-1] = nn.Tanh()
        self.cnn_module = nn.Sequential(*cnn_module)



    def forward(self, input):
        proj = self.proj(input)
        
        proj = proj.view(input.shape[0], 3 * 4, 8, 8)

        return self.cnn_module(proj)#.squeeze(1)




class features_extractor(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(features_extractor, self).__init__()
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
        self.feature_dim = pretrain_dim

        self.other = nn.ModuleDict({
            'decoder': decoder(pretrain_dim),
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

    def extract_features(self, input):
        """
        input: B x C x W x H
        output: B x D
        
        """
        DenseOut = self._getResOut(input)  # B x D
        return DenseOut

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """


        DenseOut = self._getResOut(input)  # B x D
        
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H




class TransformerFusion(nn.Module):
    def __init__(self, dim, n_class, n_layer=1, **kwargs):
        super(TransformerFusion, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(dim, dim_feedforward=1024, **kwargs)
        encoder_layer.self_attn.kdim = 32
        encoder_layer.self_attn.vdim = 32

        self.Transformer = nn.ModuleDict({
            'Encoder': nn.TransformerEncoder(encoder_layer, n_layer),
            'Attention': SimpleAttn(dim),
            'OutputLayer': nn.Sequential(nn.Linear(dim, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.Transformer['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.Transformer['OutputLayer'][0].bias, 0)


    def forward(self, input, input_num=None):
        """
        input: B x T x D
        input_num: B
        """
        B, T = input.shape[:2]
        #input = self.Transformer['projection'](input)
        EncoderIn = input.transpose(0, 1)

        if (input_num is not None):
            idxs = torch.arange(T).unsqueeze(0).repeat(B, 1)
            masks = idxs.cpu() >= input_num.unsqueeze(-1).cpu()
            masks = masks.bool().to(input.device)
            tsf_out = self.Transformer['Encoder'](
                EncoderIn, src_key_padding_mask=masks).transpose(0, 1)
        else:
            tsf_out = self.Transformer['Encoder'](EncoderIn).transpose(0, 1)

        attn = self.Transformer['Attention'](tsf_out, input_num) # B x D
        output = self.Transformer['OutputLayer'](attn) # B x 10

        return output





class features_extractor256(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(features_extractor256, self).__init__()
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
        self.feature_dim = 256

        self.other = nn.ModuleDict({
            'projection': nn.Linear(pretrain_dim, 256),
            'decoder': decoder(256),
            'OutputLayer': nn.Sequential(nn.Linear(256, n_class), nn.Sigmoid())
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

    def extract_features(self, input):
        """
        input: B x C x W x H
        output: B x D
        
        """
        DenseOut = self._getResOut(input)  # B x D
        DenseOut = self.other['projection'](DenseOut)
        return DenseOut

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """


        DenseOut = self._getResOut(input)  # B x D
        DenseOut = self.other['projection'](DenseOut)
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H


class MultiheadFusion(nn.Module):
    def __init__(self, dim, n_class, n_layer=1, **kwargs):
        super(MultiheadFusion, self).__init__()

        # encoder_layer = nn.TransformerEncoderLayer(dim, dim_feedforward=2*dim, **kwargs)
        # encoder_layer.self_attn.kdim = 32
        # encoder_layer.self_attn.vdim = 32

        self.model = nn.ModuleDict({
            'Fusion': nn.MultiheadAttention(dim, **kwargs),
            'Attention': SimpleAttn(dim),
            'OutputLayer': nn.Sequential(nn.Linear(dim, n_class), nn.Sigmoid())
        })

        nn.init.normal_(self.model['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.model['OutputLayer'][0].bias, 0)


    def forward(self, input, input_num=None):
        """
        input: B x T x D
        input_num: B
        """
        B, T = input.shape[:2]
        EncoderIn = input.transpose(0, 1)

        if (input_num is not None):
            idxs = torch.arange(T).unsqueeze(0).repeat(B, 1)
            masks = idxs.cpu() >= input_num.unsqueeze(-1).cpu()
            masks = masks.bool().to(input.device)
            tsf_out = self.model['Fusion'](
                EncoderIn, EncoderIn, EncoderIn, key_padding_mask=masks)[0].transpose(0, 1)
        else:
            tsf_out = self.model['Encoder'](EncoderIn, EncoderIn, EncoderIn)[0].transpose(0, 1)

        attn = self.model['Attention'](tsf_out, input_num) # B x D
        output = self.model['OutputLayer'](attn) # B x 10

        return output
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


class mask_pixel(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True, graph_size=256):
        super(mask_pixel, self).__init__()
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
        self.feature_dim = 512

        self.other = nn.ModuleDict({
            'mask_gen': gen_mask(graph_size),
            'projection': nn.Linear(pretrain_dim, 512),
            'decoder': decoder(512),
            'OutputLayer': nn.Sequential(nn.Linear(512, n_class), nn.Sigmoid())
        })
        nn.init.normal_(self.other['OutputLayer'][0].weight, 0, 0.1)
        nn.init.constant_(self.other['OutputLayer'][0].bias, 0)
        

    def __getResDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.ResNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.reslayer)
        return dim

    def _getResOut(self, input):
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

    def extract_features(self, input):
        """
        input: B x C x W x H
        output: B x D
        
        """
        mask = self.other['mask_gen'](input)
        mask_input = mask * input
        DenseOut = self._getResOut(mask_input)  # B x D
        DenseOut = self.other['projection'](DenseOut)
        return DenseOut

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """

        mask = self.other['mask_gen'](input)
        mask_input = mask * input
        DenseOut = self._getResOut(mask_input)  # B x D
        DenseOut = self.other['projection'](DenseOut)
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H


class extractor_256(nn.Module):
    

    def __init__(self, ResNet, n_class, Net_grad=True):
        super(extractor_256, self).__init__()
        ResNet.fc = nn.Linear(ResNet.fc.in_features, 256)
        self.ResNet = ResNet
        
        self.Net_grad = Net_grad
        
        pretrain_dim = 256
        self.feature_dim = pretrain_dim

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


        DenseOut = self.ResNet(input)  # B x D
        
        

        return self.other['OutputLayer'](DenseOut), \
            self.other['decoder'](DenseOut) # B x C x W x H
