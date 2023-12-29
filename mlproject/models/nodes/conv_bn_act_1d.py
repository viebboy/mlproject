"""
conv_bn_act_1d.py: conv1d batchnorm activation node
---------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-05-10
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""

from __future__ import annotations
import torch.nn as nn
import numpy as np
from mlproject.models.nodes.base import BaseNode
from mlproject.models.nodes.conv1d import Conv1D
from mlproject.models.nodes.batchnorm1d import BatchNorm1D
from mlproject.models.nodes.activations import build_activation_node


class ConvBnAct1D(BaseNode):
    """
    1D Convolution-BatchNorm-Activation node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        kernel_size = kwargs["kernel_size"]
        stride = kwargs["stride"]
        padding = kwargs["padding"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        groups = kwargs["groups"]
        permute_output = kwargs["permute_output"]

        if groups in [None, -1] and kernel_size != 1 and kernel_size != (1, 1):
            self.perm_indices = None
            # depth-wise separable convolution
            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    groups=in_channels,
                ),
                nn.BatchNorm1d(out_channels),
                build_activation_node(activation),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    groups=1,
                ),
                nn.BatchNorm1d(out_channels),
                build_activation_node(activation),
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
                self.perm_indices = (
                    np.transpose(np.reshape(self.perm_indices, (final_grp_value, -1)))
                    .flatten()
                    .tolist()
                )
            else:
                self.perm_indices = None

            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=final_grp_value,
                ),
                nn.BatchNorm1d(out_channels),
                build_activation_node(activation),
            )
        self.initialize(self.layers)

    @staticmethod
    def required_keys():
        return ["kernel_size", "stride", "padding", "in_channels", "out_channels"]

    @staticmethod
    def node_type():
        return "ConvBnAct1D"

    @staticmethod
    def default_values():
        return {
            **Conv1D.default_values(),
            **BatchNorm1D.default_values(),
            "permute_output": True,
        }

    def forward(self, inputs):
        outputs = self.layers(inputs)
        if self.perm_indices is not None:
            return outputs[:, self.perm_indices, :, :]
        else:
            return outputs
