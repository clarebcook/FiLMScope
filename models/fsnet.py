# these networks are modified from MVSNet 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ConvBnReLU3D, depth_regression

# num_layers - int, the number of layers in the unet 
# layer_channels - list or similar, each entry is an int specifying the number of 
# OUT channels at a specific layer. Length should equal num_layers + 1. 
# layer_strides - (x, y, z) strides to take at each layer. Shape should be (num_layers, 3) or (num_layers, 1) 
# if the stride length should be the same for x, y, and z
class CostRegNet(nn.Module):
    def __init__(self, num_channels=32, num_layers=3,
                 layer_channels=None, layer_strides=None):
        super(CostRegNet, self).__init__()
        self.num_layers = num_layers 

        # if the number of layers at each channel was not specified, generate it 
        if layer_channels is None: 
            layer_channels = np.power(2, np.arange(num_layers + 1) + 3)
        assert len(layer_channels) == num_layers + 1

        if layer_strides is None: 
            layer_strides = [[2, 2, 2]] * num_layers
        assert len(layer_strides) == num_layers

        down_layers = [ConvBnReLU3D(num_channels, layer_channels[0])] 
        channels_in = layer_channels[0]
        for strides, channels in zip(layer_strides, layer_channels[1:]): 
            conv0 = ConvBnReLU3D(channels_in, channels, stride=strides) 
            conv1 = ConvBnReLU3D(channels, channels)
            channels_in = channels 
            down_layers.append(conv0)
            down_layers.append(conv1) 
        
        up_layers = []
        for strides, channels in zip(np.flip(layer_strides, axis=0), np.flip(layer_channels, axis=0)[1:]):
            output_padding = [i - 1 for i in strides]
            conv = nn.Sequential(
                nn.ConvTranspose3d(
                    channels_in, channels, kernel_size=3, padding=1, output_padding=output_padding, stride=strides, bias=False
                ),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
            )
            channels_in = channels 
            up_layers.append(conv)
        
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)
        self.prob = nn.Conv3d(channels_in, 1, 3, stride=1, padding=1)
        return

    def forward(self, x):
        conv_outputs = [self.down_layers[0](x)]
        for num in range(self.num_layers): 
            conv0 = self.down_layers[num * 2 + 1]
            conv1 = self.down_layers[num * 2 + 2] 
            output = conv1(conv0(conv_outputs[-1]))
            conv_outputs.append(output) 

        for i, layer in enumerate(self.up_layers): 
            add_output = conv_outputs[-2 - i]
            output = add_output + layer(output) 
        
        output = self.prob(output) 
        return output

class VolumeConvNet(nn.Module):
    def __init__(self, num_channels=1, *args, **kwargs):
        super(VolumeConvNet, self).__init__()
        self.cost_regularization = CostRegNet(num_channels=num_channels, *args, **kwargs)

    def forward(self, volume_variance, depth_values):
        ## cost volume regularization
        from time import time 
        t0 = time()
        cost_reg = self.cost_regularization(volume_variance)
        t1 = time()
        cost_reg = cost_reg.squeeze(1)
        t2 = time()
        prob_volume = F.softmax(cost_reg, dim=1)
        t3 = time()
        depth = depth_regression(prob_volume, depth_values=depth_values)
        t4 = time() 

        print(t4 - t3)
        print(t3 - t2)
        print(t2 - t1)
        print(t1 - t0)

        return depth
