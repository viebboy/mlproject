"""
batchnorm2d.py: batchnorm 2d node
---------------------------------


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
from mlproject.models.nodes.base import BaseNode


class BatchNorm2D(BaseNode):
    """
    Batchnorm 2d node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

        num_features = kwargs["num_features"]
        eps = kwargs["eps"]
        momentum = kwargs["momentum"]
        affine = kwargs["affine"]
        track_running_stats = kwargs["track_running_stats"]

        self.layer = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, inputs):
        return self.layer(inputs)

    @staticmethod
    def required_keys():
        return [
            "num_features",
        ]

    @staticmethod
    def node_type():
        return "batchnorm2d"

    @staticmethod
    def default_values():
        return {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True,
        }
