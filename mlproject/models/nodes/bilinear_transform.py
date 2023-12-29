"""
bl.py: bilinear layer node
--------------------------


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
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlproject.models.nodes.base import BaseNode


def nmodeproduct(x, W, mode):
    assert mode in [1, 2], "only support mode 1, 2"
    if mode == 1:
        y = torch.transpose(x, 1, 2)
        y = F.linear(y, W)
        y = torch.transpose(y, 1, 2)
    else:
        y = F.linear(x, W)

    return y


class BilinearTransform(BaseNode):
    """
    Bilinear Transform node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]
        use_bias = kwargs["use_bias"]

        self.in1, self.in2 = input_shape
        self.out1, self.out2 = output_shape

        self.W1 = nn.Parameter(
            data=torch.Tensor(self.out1, self.in1), requires_grad=True
        )

        self.W2 = nn.Parameter(
            data=torch.Tensor(self.out2, self.in2), requires_grad=True
        )

        if use_bias:
            self.bias = nn.Parameter(
                data=torch.Tensor(1, self.out1, self.out2), requires_grad=True
            )
        else:
            self.bias = None

        self.initialize()

    def initialize(self):
        # initialization
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # 2-mode product
        y1 = nmodeproduct(x, self.W2, 2)
        # 1-mode product
        outputs = nmodeproduct(y1, self.W1, 1)

        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs

    @staticmethod
    def required_keys():
        return [
            "input_shape",
            "output_shape",
        ]

    @staticmethod
    def node_type():
        return "BilinearTransform"

    @staticmethod
    def default_values():
        return {"use_bias": True}
