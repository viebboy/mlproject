"""
conv1d.py: conv1d node
----------------------


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


class Conv1D(BaseNode):
    """
    Conv1D node
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
        groups = kwargs["groups"]
        permute_output = kwargs["permute_output"]

        assert groups in [-1, 1, None]

        if groups in [-1, None] and kernel_size != 1 and kernel_size != (1, 1):
            self.conv = nn.Conv1d(
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
                self.perm_indices = (
                    np.transpose(np.reshape(self.perm_indices, (final_grp_value, -1)))
                    .flatten()
                    .tolist()
                )
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

        self.initialize()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.perm_indices is not None:
            return outputs[:, self.perm_indices, :, :]
        else:
            return outputs

    @staticmethod
    def required_keys():
        return ["kernel_size", "stride", "padding", "in_channels", "out_channels"]

    @staticmethod
    def node_type():
        return "Conv1D"

    @staticmethod
    def default_values():
        return {
            "groups": 1,
            "bias": True,
            "activation": "silu",
            "permute_output": True,
        }
