""" Selective Kernel Networks (ResNet base)

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

This was inspired by reading 'Compounding the Performance Improvements...' (https://arxiv.org/abs/2001.06268)
and a streamlined impl at https://github.com/clovaai/assembled-cnn but I ended up building something closer
to the original paper with some modifications of my own to better balance param count vs accuracy.

Hacked together by Ross Wightman
"""

import math
import torch
from torch import nn as nn
from resnet import ResNet
from slective_kernel import SelectiveKernelConv, ConvBnAct



class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None,
                 drop_block=None, drop_path=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None):
        super(SelectiveKernelBasic, self).__init__()


        sk_kwargs = sk_kwargs or {}

        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation


        self.conv1 = SelectiveKernelConv(
            inplanes, first_planes, stride=stride, dilation=first_dilation, **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv2 = ConvBnAct(
            first_planes, outplanes, kernel_size=3, dilation=dilation, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path