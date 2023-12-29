"""
pool.py: pooling node implementation
------------------------------------


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


class MaxPool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxPool1d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            dilation=kwargs["dilation"],
            return_indices=kwargs["return_indices"],
            ceil_mode=kwargs["ceil_mode"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_pool_1d"

    @staticmethod
    def required_keys():
        return []


class MaxPool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxPool2d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            dilation=kwargs["dilation"],
            return_indices=kwargs["return_indices"],
            ceil_mode=kwargs["ceil_mode"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_pool_2d"

    @staticmethod
    def required_keys():
        return []


class MaxPool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxPool3d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            dilation=kwargs["dilation"],
            return_indices=kwargs["return_indices"],
            ceil_mode=kwargs["ceil_mode"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_pool_3d"

    @staticmethod
    def required_keys():
        return []


class MaxUnpool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxUnpool1d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
        )

    @staticmethod
    def default_values():
        return {"kernel_size": 2, "stride": None, "padding": 0}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_unpool_1d"

    @staticmethod
    def required_keys():
        return []


class MaxUnpool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxUnpool2d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
        )

    @staticmethod
    def default_values():
        return {"kernel_size": 2, "stride": None, "padding": 0}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_unpool_2d"

    @staticmethod
    def required_keys():
        return []


class MaxUnpool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.MaxUnpool3d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
        )

    @staticmethod
    def default_values():
        return {"kernel_size": 2, "stride": None, "padding": 0}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "max_unpool_3d"

    @staticmethod
    def required_keys():
        return []


class AvgPool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AvgPool1d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            ceil_mode=kwargs["ceil_mode"],
            count_include_pad=kwargs["count_include_pad"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "avg_pool_1d"

    @staticmethod
    def required_keys():
        return []


class AvgPool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AvgPool2d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            ceil_mode=kwargs["ceil_mode"],
            count_include_pad=kwargs["count_include_pad"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "avg_pool_2d"

    @staticmethod
    def required_keys():
        return []


class AvgPool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AvgPool3d(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            ceil_mode=kwargs["ceil_mode"],
            count_include_pad=kwargs["count_include_pad"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "avg_pool_3d"

    @staticmethod
    def required_keys():
        return []


class FractionalMaxPool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.FractionalMaxPool2d(
            kernel_size=kwargs["kernel_size"],
            output_size=kwargs["output_size"],
            output_ratio=kwargs["output_ratio"],
            return_indices=kwargs["return_indices"],
            _random_samples=kwargs["_random_samples"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "output_size": None,
            "output_ratio": None,
            "return_indices": False,
            "_random_samples": None,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "fractional_max_pool_2d"

    @staticmethod
    def required_keys():
        return []


class FractionalMaxPool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.FractionalMaxPool3d(
            kernel_size=kwargs["kernel_size"],
            output_size=kwargs["output_size"],
            output_ratio=kwargs["output_ratio"],
            return_indices=kwargs["return_indices"],
            _random_samples=kwargs["_random_samples"],
        )

    @staticmethod
    def default_values():
        return {
            "kernel_size": 2,
            "output_size": None,
            "output_ratio": None,
            "return_indices": False,
            "_random_samples": None,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "fractional_max_pool_3d"

    @staticmethod
    def required_keys():
        return []


class LPPool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.LPPool1d(
            norm_type=kwargs["norm_type"],
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            ceil_mode=kwargs["ceil_mode"],
        )

    @staticmethod
    def default_values():
        return {
            "norm_type": "fro",
            "kernel_size": 2,
            "stride": None,
            "ceil_mode": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "lp_pool_1d"

    @staticmethod
    def required_keys():
        return []


class LPpool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.LPPool2d(
            norm_type=kwargs["norm_type"],
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            ceil_mode=kwargs["ceil_mode"],
        )

    @staticmethod
    def default_values():
        return {
            "norm_type": "fro",
            "kernel_size": 2,
            "stride": None,
            "ceil_mode": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "lp_pool_2d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveMaxPool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveMaxPool1d(
            output_size=kwargs["output_size"],
            return_indices=kwargs["return_indices"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
            "return_indices": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_max_pool_1d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveMaxPool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveMaxPool2d(
            output_size=kwargs["output_size"],
            return_indices=kwargs["return_indices"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
            "return_indices": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_max_pool_2d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveMaxPool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveMaxPool3d(
            output_size=kwargs["output_size"],
            return_indices=kwargs["return_indices"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
            "return_indices": False,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_max_pool_3d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveAvgPool1D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveAvgPool1d(
            output_size=kwargs["output_size"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_avg_pool_1d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveAvgPool2D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveAvgPool2d(
            output_size=kwargs["output_size"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_avg_pool_2d"

    @staticmethod
    def required_keys():
        return []


class AdaptiveAvgPool3D(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.AdaptiveAvgPool3d(
            output_size=kwargs["output_size"],
        )

    @staticmethod
    def default_values():
        return {
            "output_size": 1,
        }

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "adaptive_avg_pool_3d"

    @staticmethod
    def required_keys():
        return []


POOL_NODES = {
    AvgPool1D.node_type(): AvgPool1D,
    AvgPool2D.node_type(): AvgPool2D,
    AvgPool3D.node_type(): AvgPool3D,
    MaxPool1D.node_type(): MaxPool1D,
    MaxPool2D.node_type(): MaxPool2D,
    MaxPool3D.node_type(): MaxPool3D,
    FractionalMaxPool2D.node_type(): FractionalMaxPool2D,
    FractionalMaxPool3D.node_type(): FractionalMaxPool3D,
    LPPool1D.node_type(): LPPool1D,
    LPpool2D.node_type(): LPpool2D,
    AdaptiveMaxPool1D.node_type(): AdaptiveMaxPool1D,
    AdaptiveMaxPool2D.node_type(): AdaptiveMaxPool2D,
    AdaptiveMaxPool3D.node_type(): AdaptiveMaxPool3D,
    AdaptiveAvgPool1D.node_type(): AdaptiveAvgPool1D,
    AdaptiveAvgPool2D.node_type(): AdaptiveAvgPool2D,
    AdaptiveAvgPool3D.node_type(): AdaptiveAvgPool3D,
}
