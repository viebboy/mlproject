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


class BaseNode(nn.Module):
    def __init__(self, **kwargs: dict):
        super().__init__()

        for key in self.required_keys():
            if key not in kwargs:
                msg = ''.join([
                    f'missing required parameter {key} in the configuration of the following node',
                    f'(node type: {self.node_type()}, node name: {kwargs["name"]})'
                ])
                logger.error(msg)
                raise ValueError(msg)

        self.merge_with_default(kwargs)

    def merge_with_default(self, kwargs: dict):
        for key, value in self.default_values().items():
            if key not in kwargs:
                kwargs[key] = value

    def required_keys(self):
        raise NotImplemented

    def node_type(self):
        raise NotImplemented

    def default_values(self):
        raise NotImplemented


class Sum(BaseNode):
    """
    Summation node
    """
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        x = 0
        for item in inputs:
            x += item
        return x

    def required_keys(self):
        return []

    def node_type(self):
        return 'Sum'

    def default_values(self):
        raise {}


class Concatenate(BaseNode):
    """
    Concatenation node
    """
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        self.axis = kwargs['axis']

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.axis)

    def required_keys(self):
        return ['axis']

    def node_type(self):
        return 'Concatenate'

    def default_values(self):
        raise {}


class Conv2D(BaseNode):
    """
    Conv2D node
    """
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)

        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        padding = kwargs['padding']
        bias = kwargs['bias']
        activation = kwargs['activation']
        groups = kwargs['groups']
        permute_output = kwargs['permute_output']

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
            self.perm_indices = None
        else:
            if groups is None:
                groups = 1

            # auto infer the groups value that is divisble by both in_channels
            # and out_channels
            final_grp_value = None
            for grp_value in range(groups, 0, -1):
                if (in_channels % grp_value == 0) and (out_channels % grp_value == 0):
                    final_grp_value = grp_value
                    break

            if permute_output and final_grp_value > 1:
                self.perm_indices = np.arange(out_channels)
                self.perm_indices = np.transpose(
                    np.reshape(self.perm_indices, (final_grp_value, -1))
                ).flatten().tolist()
            else:
                self.perm_indices = None

            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=final_grp_value,
            )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.perm_indices is not None:
            return outputs[:, self.perm_indices, :, :]
        else:
            return outputs

    def required_keys(self):
        return [
            'kernel_size',
            'stride',
            'padding',
            'in_channels',
            'out_channels'
        ]

    def node_type(self):
        return 'Conv2D'

    def default_values(self):
        return {
            'groups': 1,
            'bias': True,
            'activation': 'silu',
            'permute_output': True,
        }


class ConvBnAct2D(BaseNode):
    """
    2D Convolution-BatchNorm-Activation node
    """
    def __init__(**kwargs: dict):
        super().__init__(**kwargs)

        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        padding = kwargs['padding']
        bias = kwargs['bias']
        activation = kwargs['activation']
        groups = kwargs['groups']
        permute_output = kwargs['permute_output']

        if groups in [None, -1] and kernel_size != 1 and kernel_size != (1, 1):
            self.perm_indices = None
            # depth-wise separable convolution
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
            # for kernel_size is 1, we don't want to build depth-wise conv
            if groups in [None, -1]:
                groups = 1

            # auto infer the groups value that is divisble by both in_channels
            # and out_channels
            final_grp_value = None
            for grp_value in range(groups, 0, -1):
                if (in_channels % grp_value == 0) and (out_channels % grp_value == 0):
                    final_grp_value = grp_value
                    break

            if permute_output and final_grp_value > 1:
                self.perm_indices = np.arange(out_channels)
                self.perm_indices = np.transpose(
                    np.reshape(self.perm_indices, (final_grp_value, -1))
                ).flatten().tolist()
            else:
                self.perm_indices = None

            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=final_grp_value,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )

    def required_keys(self):
        return [
            'kernel_size',
            'stride',
            'padding',
            'in_channels',
            'out_channels'
        ]

    def node_type(self):
        return 'ConvBnAct2D'

    def forward(self, inputs):
        outputs = self.layers(inputs)
        if self.perm_indices is not None:
            return outputs[:, self.perm_indices, :, :]
        else:
            return outputs


def get_supported_node_names():
    return

def build_node(node_config: dict):
    return
