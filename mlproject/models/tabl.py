"""
tabl.py: TABL model family
--------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def nmodeproduct(x, W, mode):
    assert mode in [1, 2], "only support mode 1, 2"
    if mode == 1:
        y = torch.transpose(x, 1, 2)
        y = F.linear(y, W)
        y = torch.transpose(y, 1, 2)
    else:
        y = F.linear(x, W)

    return y


class BilinearLayer(nn.Module):
    """
    Bilinear Layer
    """

    def __init__(self, input_shape, output_shape, use_bias=True):
        super(BilinearLayer, self).__init__()
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


class BilinearModel(nn.Module):
    """
    Bilinear Model
    """

    def __init__(
        self,
        input_shape,
        hidden_shapes,
        activation=nn.ReLU(),
        use_bias=True,
    ):
        super(BilinearModel, self).__init__()

        self.all_layers = nn.ModuleList()
        for idx, hidden_shape in enumerate(hidden_shapes):
            self.all_layers.append(
                BilinearLayer(input_shape, hidden_shape, use_bias=use_bias)
            )
            if idx < len(hidden_shapes) - 1 and activation is not None:
                self.all_layers.append(activation)
            input_shape = hidden_shape

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)

        return x


class BiN(nn.Module):
    """
    Bilinear Normalization Layer
    """

    def __init__(
        self,
        input_shape,
        epsilon=1e-4,
    ):
        super(BiN, self).__init__()

        self.dim1, self.dim2 = input_shape
        self.epsilon = epsilon

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


class TABLLayer(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        attention_axis,
        use_bias=True,
    ):
        super(TABLLayer, self).__init__()

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

        # initialization
        nn.init.xavier_uniform_(self.W1)
        nn.init.constant_(self.W, 1.0 / self.attention_dim)
        nn.init.xavier_uniform_(self.W2)
        nn.init.constant_(self.alpha, 0.5)
        if use_bias:
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


class TABLModel(nn.Module):
    """
    TABL Model: all layers are BilinearLayer, except the last layer, which is TABLLayer
    """

    def __init__(
        self,
        input_shape,
        hidden_shapes,
        attention_axis,
        activation,
        use_bias=True,
    ):
        super(TABLModel, self).__init__()

        self.all_layers = nn.ModuleList()
        for idx, hidden_shape in enumerate(hidden_shapes[:-1]):
            self.all_layers.append(BilinearLayer(input_shape, hidden_shape, use_bias))
            if activation is not None:
                self.all_layers.append(activation)
            input_shape = hidden_shape

        self.all_layers.append(
            TABLLayer(input_shape, hidden_shapes[-1], attention_axis, use_bias)
        )

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)

        return x
