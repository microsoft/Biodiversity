# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Optional

from . import dla as dla_modules


class OWLC_Architecture(nn.Module):
    ''' OWL-C (Overhead Wildlife Locator - CNN) architecture using only the localization branch '''

    def __init__(
        self,
        num_layers: int = 34,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16. 
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end 
                of decoder. Defaults to 64.
        '''
        super(OWLC_Architecture, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)

        self.down_ratio = down_ratio
        self.head_conv = head_conv

        self.first_level = int(np.log2(down_ratio))

        # backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )

        # localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                ),
            nn.Sigmoid()
            )

        self.loc_head[-2].bias.data.fill_(0.00)
        
    def forward(self, input: torch.Tensor):
        encode = self.base_0(input)    
        bottleneck = self.bottleneck_conv(encode[-1])
        encode[-1] = bottleneck
        decode_hm = self.dla_up(encode[self.first_level:])
        heatmap = self.loc_head(decode_hm)

        return heatmap
    
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    