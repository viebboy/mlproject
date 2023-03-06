"""
densenet1d.py: DenseNet model for timeseries
--------------------------------------------


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


def get_positional_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    P = np.expand_dims(np.transpose(P), 0).astype('float32')
    return P

class PositionalEncoding(nn.Module):
    def __init__(self, input_len, input_features):
        super().__init__()
        encoder = get_positional_encoding(input_len, input_features)
        self.register_buffer('positional_shift', torch.Tensor(encoder))

    def forward(self, x):
        return x + self.positional_shift

class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation,
        use_batchnorm,
        groups,
    ):
        super().__init__()
        if use_batchnorm:
            self.add_module('bn', nn.BatchNorm1d(in_channels))

        self.add_module('activation', activation)

        self.add_module(
            'conv',
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                groups=groups,
            )
        )
        self.initialize()

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        nb_prev_layers,
        growth_rate,
        bottleneck,
        activation,
        use_batchnorm,
        groups,
        dropout,
    ):
        super().__init__()

        if groups is not None and groups > 1:
            # find group value for 1st conv
            grp1 = None
            for grp in range(groups, 0, -1):
                if (in_channels + nb_prev_layers) % grp == 0 and (int(growth_rate * bottleneck) % grp ==0):
                    grp1 = grp
                    break
            if grp1 is None:
                grp1 = 1

            # find group value for 2nd conv
            grp2 = None
            for grp in range(groups, 0, -1):
                if (int(growth_rate * bottleneck) % grp ==0) and growth_rate % grp == 0:
                    grp2 = grp
                    break

            if grp2 is None:
                grp2 = 1
        else:
            grp1 = 1
            grp2 = 1

        if grp1 == 1:
            self.perm_indices1 = None
        else:
            self.perm_indices1 = np.arange(int(bottleneck * growth_rate))
            self.perm_indices1 = np.transpose(np.reshape(self.perm_indices1, (grp1, -1))).flatten().tolist()

        if grp2 == 1:
            self.perm_indices2 = None
        else:
            self.perm_indices2 = np.arange(growth_rate)
            self.perm_indices2 = np.transpose(np.reshape(self.perm_indices2, (grp1, -1))).flatten().tolist()

        self.conv1 = Conv(
            in_channels=in_channels + nb_prev_layers,
            out_channels=int(bottleneck * growth_rate),
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
            use_batchnorm=use_batchnorm,
            groups=grp1,
        )

        self.conv2 = Conv(
            in_channels=int(growth_rate * bottleneck),
            out_channels=growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation,
            use_batchnorm=use_batchnorm,
            groups=grp2,
        )

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv1(x)
        if self.perm_indices1 is not None:
            x = x[:,self.perm_indices1, :]
        x = self.conv2(x)
        if self.perm_indices2 is not None:
            x = x[:,self.perm_indices2, :]

        if self.dropout is not None:
            x = self.dropout(x)

        return x

class DenseBlock(nn.Module):
    def __init__(
        self,
        nb_layers,
        in_channels,
        growth_rate,
        bottleneck,
        activation,
        use_batchnorm,
        groups,
        dropout,
        reduce_input,
        positional_encoding,
        input_len,
    ):
        super().__init__()
        self.reduce_input = reduce_input

        if positional_encoding:
            assert input_len is not None
            self.positional_encoder = PositionalEncoding(input_len, in_channels)
        else:
            self.positional_encoder = None

        if reduce_input:
            self.first_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=growth_rate,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            block_in_channels = growth_rate
        else:
            block_in_channels = in_channels

        self.dense_layers = nn.ModuleList()
        nb_prev_layers = 0

        for i in range(nb_layers):
            self.dense_layers.append(
                DenseLayer(
                    in_channels=block_in_channels,
                    nb_prev_layers=nb_prev_layers,
                    growth_rate=growth_rate,
                    bottleneck=bottleneck,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                    groups=groups,
                    dropout=dropout,
                )
            )
            nb_prev_layers += growth_rate

        self.output_channels = block_in_channels + nb_layers * growth_rate

    def forward(self, x):
        if self.positional_encoder is not None:
            x = self.positional_encoder(x)

        if self.reduce_input:
            x = self.first_conv(x)

        for layer in self.dense_layers:
            out = layer(x)
            x = torch.cat([out, x], dim=1)

        return x

class GlobalAverage(nn.Module):
    def forward(self, x):
        # perform averaging in time dimensions
        return x.mean(-1)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return x


class DenseNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        nb_features = kwargs['nb_features']
        nb_init_filters = kwargs['nb_init_filters']
        init_kernel_size = kwargs['init_kernel_size']
        stages = kwargs['stages']
        growth_rates = kwargs['growth_rates']
        bottleneck = kwargs['bottleneck']
        activation = kwargs['activation']
        use_batchnorm = kwargs['use_batchnorm']
        groups = kwargs['groups']
        dropout = kwargs['dropout']
        reduce_input = kwargs['reduce_input']
        input_len = kwargs['input_len']
        input_positional_encoding = kwargs['input_positional_encoding']
        intermediate_positional_encoding = kwargs['intermediate_positional_encoding']
        global_average = kwargs['global_average']
        embedding_dim = kwargs['embedding_dim']
        use_bias_for_embedding_layer = kwargs['use_bias_for_embedding_layer']

        assert len(stages) == len(growth_rates)

        if input_positional_encoding:
            assert input_len is not None
            self.positional_encoder = PositionalEncoding(input_len, nb_features)
        else:
            self.positional_encoder = None

        # first conv
        # this layer's supposed to reduce time dimension by half
        self.first_conv = nn.Conv1d(
            in_channels=nb_features,
            out_channels=nb_init_filters,
            kernel_size=init_kernel_size,
            stride=2,
            padding=1,
            bias=True,
        )

        if input_len is not None:
            input_len = int(np.ceil(input_len/2))

        self.hidden_layers = nn.ModuleList()
        in_channels = nb_init_filters

        for idx, (stage, growth_rate) in enumerate(zip(stages, growth_rates)):
            block = DenseBlock(
                nb_layers=stage,
                in_channels=in_channels,
                growth_rate=growth_rate,
                bottleneck=bottleneck,
                activation=activation,
                use_batchnorm=use_batchnorm,
                groups=groups,
                dropout=dropout,
                reduce_input=reduce_input,
                positional_encoding=intermediate_positional_encoding,
                input_len=input_len
            )
            in_channels = block.output_channels
            self.hidden_layers.append(block)

            if idx != len(stages) - 1:
                # not the last block
                # append trasition layer
                self.hidden_layers.append(nn.MaxPool1d(kernel_size=2))
                if input_len is not None:
                    input_len = int(np.floor(input_len / 2))

        if not global_average:
            in_channels = in_channels * input_len

        if global_average:
            self.hidden_layers.append(GlobalAverage())
        else:
            self.hidden_layers.append(Flatten())

        self.embedding_layer = nn.Linear(
            in_features=in_channels,
            out_features=embedding_dim,
            bias=use_bias_for_embedding_layer,
        )

        self.initialize()

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        if self.positional_encoder is not None:
            x = self.positional_encoder(x)

        x = self.first_conv(x)
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.embedding_layer(x)
        return x

if __name__ == '__main__':
    params = {
        'nb_features': 10,
        'nb_init_filters': 24,
        'init_kernel_size': 3,
        'stages': [3, 3, 3],
        'growth_rates': [12, 12, 12],
        'bottleneck': 0.5,
        'activation': torch.nn.SiLU(inplace=True),
        'use_batchnorm': True,
        'groups': 2,
        'dropout': 0.2,
        'reduce_input': False,
        'input_positional_encoding': True,
        'intermediate_positional_encoding': True,
        'input_len': 100,
        'embedding_dim': 128,
        'global_average': False,
        'use_bias_for_embedding_layer': True,
    }

    net = DenseNet(**params)

    x = torch.rand(1, params['nb_features'], params['input_len'])
    print(net)
    with torch.no_grad():
        y = net(x)
        print(y.shape)
