"""
models.py: model architecture implementation
--------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-02-13
* Version: 0.0.1

This is part of the cifar10 example project

License
-------
Apache 2.0 License

"""

import torch
import torch.nn as nn

from mlproject.models.densenet2d import DenseNet as DenseNetBackbone


class DenseNet(nn.Module):
    """
    DenseNet classifier using the backbone implementation from mlproject.models.densenet2d.DenseNet
    """

    def __init__(self, **kwargs):
        super().__init__()
        # create the backbone
        self.backbone = DenseNetBackbone(**kwargs)
        # create prediction layer
        self.classifier = nn.Linear(
            in_features=kwargs["embedding_dim"],
            out_features=kwargs["nb_class"],
            bias=kwargs["use_bias_for_output_layer"],
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # here we implement a sample config and test if our model works
    sample_config = {
        "nb_init_filters": 24,
        "growth_rates": [(16, 16), (16, 16), (16, 16), (16, 16)],
        "bottleneck": 0.5,
        "bottleneck_style": "in",
        "activation": nn.SiLU(inplace=True),
        "groups": 2,
        "reduce_input": False,
        "dropout": None,
        "pool_in_last_block": True,
        "global_average": True,
        "use_bias_for_embedding_layer": False,
        "embedding_dim": 512,
        "input_height": 256,
        "input_width": 128,
        "use_bias_for_output_layer": True,
        "nb_class": 10,
    }
    net = DenseNet(**sample_config)

    x = torch.rand(1, 3, 256, 128)
    with torch.no_grad():
        y = net(x)
    print(net)
