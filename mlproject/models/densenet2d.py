"""
densenet2d.py: densenet implementation for image
------------------------------------------------


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
import thop
from loguru import logger





class Conv(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()

        # get parameters
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        padding = kwargs['padding']
        activation = kwargs['activation']
        groups = kwargs['groups']

        if 'bias' in kwargs.keys():
            bias = kwargs['bias']
        else:
            bias = False

        if 'permute_output' in kwargs.keys():
            permute_output = kwargs['permute_output']
        else:
            permute_output = False


        if permute_output and groups > 1:
            self.perm_indices = np.arange(out_channels)
            self.perm_indices = np.transpose(np.reshape(self.perm_indices, (groups, -1))).flatten().tolist()
        else:
            self.perm_indices = None

        self.add_module('bn', nn.BatchNorm2d(in_channels))

        self.add_module('activation', activation)

        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        )

        self.initialize()

    def forward(self, x):
        x = super(Conv, self).forward(x)
        if self.perm_indices is not None:
            x = x[:, self.perm_indices, :, :]
        return x

    def count_params(self):
        total = 0
        total += self.conv.weight.numel()
        if self.conv.bias is not None:
            total += self.conv.bias.numel()
        if self.perm_indices is not None:
            total += len(self.perm_indices)
        return total


    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    nn.init.constant_(layer.bias, 0.0)


class DenseLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # get parameters
        in_channels = kwargs['in_channels']
        bottleneck = kwargs['bottleneck']
        bottleneck_style = kwargs['bottleneck_style']
        out_channels = kwargs['out_channels']
        activation = kwargs['activation']
        groups = kwargs['groups']
        dropout = kwargs['dropout']

        if bottleneck_style == 'out':
            bottleneck_dim = int(bottleneck * out_channels)
        elif bottleneck_style == 'in':
            bottleneck_dim = int(bottleneck * in_channels)
        else:
            raise RuntimeError(f'unknown bottleneck_style={bottleneck_style}')

        if groups is not None and groups > 1:
            # find group value for 1st conv
            nb_group1 = None
            for grp in range(groups, 0, -1):
                if in_channels % grp == 0 and (bottleneck_dim % grp ==0):
                    nb_group1 = grp
                    break
            if nb_group1 is None:
                nb_group1 = 1
        else:
            nb_group1 = 1

        self.carry_indices = None

        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=bottleneck_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
            groups=nb_group1,
            permute_output=True,
        )

        self.conv2 = Conv(
            in_channels=bottleneck_dim,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation,
            groups=1,
        )

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self._input_channels = in_channels

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        # select which channels to carry forward
        if self.carry_indices is not None:
            x = x[:, self.carry_indices, :, :]
        # then concatenate them with the output
        y = torch.cat([y, x], dim=1)

        if self.dropout is not None:
            y = self.dropout(y)

        return y

    def get_input_channels(self):
        return self._input_channels

    def set_carry_indices(self, carry_indices):
        assert max(carry_indices) <= self._input_channels - 1
        self.carry_indices = carry_indices

    def count_params(self):
        total = (
            self.conv1.count_params() +
            self.conv2.count_params()
        )

        return total


class DenseBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # get parameters
        in_channels = kwargs['in_channels']
        bottleneck = kwargs['bottleneck']
        bottleneck_style = kwargs['bottleneck_style']
        growth_rates = kwargs['growth_rates']
        activation = kwargs['activation']
        groups = kwargs['groups']
        dropout = kwargs['dropout']
        reduce_input = kwargs['reduce_input']

        if reduce_input:
            self.first_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=growth_rates[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            block_in_channels = growth_rates[0]
        else:
            self.first_conv = None
            block_in_channels = in_channels

        self.dense_layers = nn.ModuleList()
        nb_prev_layers = 0

        for growth_rate in growth_rates:
            self.dense_layers.append(
                DenseLayer(
                    in_channels=block_in_channels,
                    bottleneck=bottleneck,
                    bottleneck_style=bottleneck_style,
                    out_channels=growth_rate,
                    activation=activation,
                    groups=groups,
                    dropout=dropout,
                )
            )
            block_in_channels += growth_rate

        self._output_channels = block_in_channels

        self.initialize()

    def forward(self, x):
        if self.first_conv is not None:
            x = self.first_conv(x)

        for layer in self.dense_layers:
            x = layer(x)

        return x

    def get_output_channels(self):
        return self._output_channels

    def count_params(self):
        total = 0
        for layer in self.dense_layers:
            total += layer.count_params()
        if self.first_conv is not None:
            total += self.first_conv.weight.numel()

        return total

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)


class GlobalAverage(nn.Module):
    def forward(self, x):
        # perform averaging in height and width dimensions
        return x.mean(-1).mean(-1)

class Flatten(nn.Module):
    def forward(self, x):
        # perform averaging in height and width dimensions
        x = x.reshape(x.size(0), -1)
        return x


class DenseNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # get parameters
        nb_init_filters = kwargs['nb_init_filters']
        bottleneck = kwargs['bottleneck']
        bottleneck_style = kwargs['bottleneck_style']
        growth_rates = kwargs['growth_rates']
        activation = kwargs['activation']
        groups = kwargs['groups']
        dropout = kwargs['dropout']
        reduce_input = kwargs['reduce_input']
        pool_in_last_block = kwargs['pool_in_last_block']
        embedding_dim = kwargs['embedding_dim']
        global_average = kwargs['global_average']
        use_bias_for_embedding_layer = kwargs['use_bias_for_embedding_layer']
        input_height = kwargs['input_height']
        input_width = kwargs['input_width']
        self._input_height = input_height
        self._input_width = input_width

        # first conv
        self.first_conv = nn.Conv2d(
            in_channels=12,
            out_channels=nb_init_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.hidden_layers = nn.ModuleList()
        in_channels = nb_init_filters

        for idx, growth_rate_block in enumerate(growth_rates):
            if idx == 0:
                reduce_input_ = False
            else:
                reduce_input_ = reduce_input

            block = DenseBlock(
                in_channels=in_channels,
                bottleneck=bottleneck,
                bottleneck_style=bottleneck_style,
                growth_rates=growth_rate_block,
                activation=activation,
                groups=groups,
                dropout=dropout,
                reduce_input=reduce_input_,
            )
            in_channels = block.get_output_channels()
            self.hidden_layers.append(block)

            if pool_in_last_block:
                self.hidden_layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                if idx != len(growth_rates) - 1:
                    self.hidden_layers.append(nn.MaxPool2d(kernel_size=2))

        if global_average:
            self.hidden_layers.append(GlobalAverage())
        else:
            self.hidden_layers.append(Flatten())

        # compute number of channels before generating embedding
        with torch.no_grad():
            x = torch.rand(1, 3, input_height, input_width)
            # rearrange
            patch_top_left = x[..., ::2, ::2]
            patch_top_right = x[..., ::2, 1::2]
            patch_bot_left = x[..., 1::2, ::2]
            patch_bot_right = x[..., 1::2, 1::2]
            x = torch.cat(
                (
                    patch_top_left,
                    patch_bot_left,
                    patch_top_right,
                    patch_bot_right,
                ),
                dim=1,
            )

            x = self.first_conv(x)
            for layer in self.hidden_layers:
                x = layer(x)

        # add final FC layer to generating embedding of fixed dimension
        self.hidden_layers.append(
            nn.Linear(
                in_features=x.size(1),
                out_features=embedding_dim,
                bias=use_bias_for_embedding_layer
            )
        )
        self.initialize()

    def rearrange_input(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.first_conv(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return x

    def complexity(self):
        with torch.no_grad():
            x = torch.rand(1, 3, self._input_height, self._input_width)
            flops, params = thop.profile(self, inputs=(x,))
        logger.info('This model has {:.2f} M parameters and {:.2f} GFLOPs'.format(params/1e6, flops/1e9))
        return params, flops

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)



if __name__ == '__main__':
    sample_config = {
        'nb_init_filters': 24,
        'growth_rates': [(16, 16), (16, 16), (16, 16), (16, 16)],
        'bottleneck': 0.5,
        'bottleneck_style': 'in',
        'activation': nn.SiLU(inplace=True),
        'groups': 2,
        'reduce_input': False,
        'dropout': None,
        'pool_in_last_block': True,
        'global_average': True,
        'use_bias_for_embedding_layer': True,
        'embedding_dim': 512,
        'input_height': 256,
        'input_width': 128,
    }
    net = DenseNet(**sample_config)

    x = torch.rand(1, 3, 256, 128)
    with torch.no_grad():
        y = net(x)
    print(net)
    logger.info(f'output from net has shape: {y.shape}')
    net.complexity()
