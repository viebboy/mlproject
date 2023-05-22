"""
nodes.py: network nodes
------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-05-10
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU(inplace=True)
    elif name == "relu":
        module = nn.ReLU(inplace=True)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=True)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Sum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        x = 0
        for item in inputs:
            x += item
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, groups):
        super().__init__()
        assert groups in [-1, 1, None]
        if groups in [-1, None] and kernel_size != 1 and kernel_size != (1, 1):
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=in_channels,
            )
        else:
            if groups is None:
                groups = 1
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def forward(self, inputs):
        return self.conv(inputs)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, groups):
        super().__init__()

        if groups in [None, -1] and kernel_size != 1 and kernel_size != (1, 1):
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    groups=1,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )
        else:
            if groups in [None, -1]:
                groups = 1
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )

    def forward(self, inputs):
        return self.layers(inputs)
