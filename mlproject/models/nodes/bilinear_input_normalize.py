"""
bilinear_input_normalize.py: bilinear input normalization node
--------------------------------------------------------------


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
from mlproject.models.nodes.base import BaseNode


class BilinearInputNormalize(BaseNode):
    """
    Bilinear Input Normalization node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

        self.dim1, self.dim2 = kwargs["input_shape"]
        self.epsilon = kwargs["epsilon"]

        self.gamma1 = nn.Parameter(
            data=torch.Tensor(1, self.dim1, 1),
            requires_grad=True,
        )

        self.beta1 = nn.Parameter(
            data=torch.Tensor(1, self.dim1, 1),
            requires_grad=True,
        )

        self.gamma2 = nn.Parameter(
            data=torch.Tensor(1, 1, self.dim2),
            requires_grad=True,
        )

        self.beta2 = nn.Parameter(
            data=torch.Tensor(1, 1, self.dim2),
            requires_grad=True,
        )

        self.lambda1 = nn.Parameter(
            data=torch.Tensor(
                1,
            ),
            requires_grad=True,
        )

        self.lambda2 = nn.Parameter(
            data=torch.Tensor(
                1,
            ),
            requires_grad=True,
        )

        self.initialize()

    def initialize(self):
        # initialization
        nn.init.ones_(self.gamma1)
        nn.init.zeros_(self.beta1)
        nn.init.ones_(self.gamma2)
        nn.init.zeros_(self.beta2)
        nn.init.ones_(self.lambda1)
        nn.init.ones_(self.lambda2)

    def forward(self, x):
        # normalize temporal mode
        # N x T x D ==> N x 1 x D or
        # N x D x T ==> N x D x 1.
        dim1_mean = torch.mean(x, 1, keepdims=True)

        # N x T x D ==> N x 1 x D or
        # N x D x T ==> N x D x 1.
        dim1_std = torch.std(x, 1, keepdims=True)

        # mask = tem_std >= self.epsilon
        # tem_std = tem_std*mask + torch.logical_not(mask)*torch.ones(tem_std.size(), requires_grad=False)
        dim1_std[dim1_std < self.epsilon] = 1.0
        dim1 = (x - dim1_mean) / dim1_std

        # N x T x D ==> N x T x 1 or
        # N x D x T ==> N x 1 x T.
        dim2_mean = torch.mean(x, 2, keepdims=True)
        dim2_std = torch.std(x, 2, keepdims=True)

        dim2_std[dim2_std < self.epsilon] = 1.0
        dim2 = (x - dim2_mean) / dim2_std

        outputs1 = self.gamma1 * dim1 + self.beta1
        outputs2 = self.gamma2 * dim2 + self.beta2

        return self.lambda1 * outputs1 + self.lambda2 * outputs2

    @staticmethod
    def required_keys():
        return [
            "input_shape",
        ]

    @staticmethod
    def node_type():
        return "BiN"

    @staticmethod
    def default_values():
        return {"epsilon": 1e-6}
