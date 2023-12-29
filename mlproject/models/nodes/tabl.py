"""
tabl.py: temporal attention bilinear transform node
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class TABL(BaseNode):
    """
    Temporal Attention Bilinear Transform node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

        input_shape = kwargs["input_shape"]
        output_shape = kwargs["output_shape"]
        use_bias = kwargs["use_bias"]
        attention_axis = kwargs["attention_axis"]

        self.in1, self.in2 = input_shape
        self.out1, self.out2 = output_shape

        self.W1 = nn.Parameter(
            data=torch.Tensor(self.out1, self.in1), requires_grad=True
        )  # D' x D.

        self.attention_axis = attention_axis
        if attention_axis in [2, -1]:
            self.W = nn.Parameter(data=torch.Tensor(self.in2, self.in2))  # T x T.
            self.attention_dim = self.in2
        else:
            self.W = nn.Parameter(data=torch.Tensor(self.in1, self.in1))  # D x D.
            self.attention_dim = self.in1

        self.register_buffer("I", torch.tensor(np.eye(self.attention_dim)))

        self.W2 = nn.Parameter(
            data=torch.Tensor(self.out2, self.in2), requires_grad=True
        )  # T' x T.

        self.alpha = nn.Parameter(
            data=torch.Tensor(
                1,
            ),
            requires_grad=True,
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
        nn.init.constant_(self.W, 1.0 / self.attention_dim)
        nn.init.xavier_uniform_(self.W2)
        nn.init.constant_(self.alpha, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.attention_axis in [-1, 2]:
            x_bar = nmodeproduct(x, self.W1, 1)
            W = self.W - self.W * self.I + self.I / float(self.attention_dim)
            E = nmodeproduct(x_bar, W.float(), 2)
            A = F.softmax(E, dim=-1)
            alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
            x_tilde = alpha * (x_bar * A) + (1 - alpha) * x_bar
            y = nmodeproduct(x_tilde, self.W2, 2)
            if self.bias is not None:
                y = y + self.bias
        else:
            x_bar = nmodeproduct(x, self.W2, 2)
            W = self.W - self.W * self.I + self.I / float(self.attention_dim)
            E = nmodeproduct(x_bar, W.float(), 1)
            A = F.softmax(E, dim=1)
            alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
            x_tilde = alpha * (x_bar * A) + (1 - alpha) * x_bar
            y = nmodeproduct(x_tilde, self.W1, 1)
            if self.bias is not None:
                y = y + self.bias

        return y

    def required_keys(self):
        return [
            "input_shape",
            "output_shape",
            "attention_axis",
        ]

    @staticmethod
    def node_type():
        return "TABL"

    @staticmethod
    def default_values():
        return {"use_bias": True}
