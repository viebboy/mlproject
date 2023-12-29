"""
activations.py: activation node implementation
----------------------------------------------


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
from typing import Union
from mlproject.models.nodes.base import BaseNode


class Identity(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)

    @staticmethod
    def required_keys():
        return []

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return x

    @staticmethod
    def node_type():
        return "identity"


class ELU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.ELU(alpha=kwargs["alpha"], inplace=kwargs["inplace"])

    @staticmethod
    def required_keys():
        return []

    @staticmethod
    def default_values():
        return {"alpha": 1.0, "inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "ELU"


class Hardshrink(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Hardshrink(lambd=kwargs["lambd"])

    @staticmethod
    def default_values():
        return {"lambd": 0.5}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "hard_shrink"

    @staticmethod
    def required_keys():
        return []


class Hardsigmoid(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Hardsigmoid(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "hard_sigmoid"

    @staticmethod
    def required_keys():
        return []


class Hardtanh(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Hardtanh(
            min_val=kwargs["min_val"],
            max_val=kwargs["max_val"],
            inplace=kwargs["inplace"],
        )

    @staticmethod
    def default_values():
        return {"min_val": -1.0, "max_val": 1.0, "inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "hard_tanh"

    @staticmethod
    def required_keys():
        return []


class Hardswish(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Hardswish(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "hard_swish"

    @staticmethod
    def required_keys():
        return []


class LogSigmoid(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.LogSigmoid()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "log_sigmoid"

    @staticmethod
    def required_keys():
        return []


class PReLU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.PReLU(num_parameters=1, init=kwargs["init"])

    @staticmethod
    def default_values():
        return {"init": 0.25}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "prelu"

    @staticmethod
    def required_keys():
        return []


class ReLU6(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.ReLU6(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "relu6"

    @staticmethod
    def required_keys():
        return []


class RReLU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.RReLU(
            inplace=kwargs["inplace"],
            lower=kwargs["lower"],
            upper=kwargs["upper"],
        )

    @staticmethod
    def default_values():
        return {"inplace": True, "lower": 0.125, "upper": 0.3333333333333333}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "rrelu"

    @staticmethod
    def required_keys():
        return []


class SELU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.SELU(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "selu"

    @staticmethod
    def required_keys():
        return []


class CELU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.CELU(alpha=kwargs["alpha"], inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"alpha": 1.0, "inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "celu"

    @staticmethod
    def required_keys():
        return []


class GELU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.GELU()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "gelu"

    @staticmethod
    def required_keys():
        return []


class Sigmoid(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Sigmoid()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "sigmoid"

    @staticmethod
    def required_keys():
        return []


class Softplus(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Softplus(beta=kwargs["beta"], threshold=kwargs["threshold"])

    @staticmethod
    def default_values():
        return {"beta": 1, "threshold": 20}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "soft_plus"

    @staticmethod
    def required_keys():
        return []


class Softshrink(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Softshrink(lambd=kwargs["lambd"])

    @staticmethod
    def default_values():
        return {"lambd": 0.5}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "soft_shrink"

    @staticmethod
    def required_keys():
        return []


class Softsign(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Softsign()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "soft_sign"

    @staticmethod
    def required_keys():
        return []


class Tanh(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Tanh()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "tanh"

    @staticmethod
    def required_keys():
        return []


class Tanhshrink(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Tanhshrink()

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "tanh_shrink"

    @staticmethod
    def required_keys():
        return []


class Threshold(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Threshold(threshold=kwargs["threshold"], value=kwargs["value"])

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "threshold"

    @staticmethod
    def required_keys():
        return ["threshold", "value"]


class ReLU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.ReLU(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "relu"

    @staticmethod
    def required_keys():
        return []


class LeakyReLU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.LeakyReLU(
            negative_slope=kwargs["negative_slope"], inplace=kwargs["inplace"]
        )

    @staticmethod
    def default_values():
        return {"negative_slope": 0.01, "inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "leaky_relu"

    @staticmethod
    def required_keys():
        return []


class SiLU(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.SiLU(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {"inplace": True}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "silu"

    @staticmethod
    def required_keys():
        return []


class Mish(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Mish(inplace=kwargs["inplace"])

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "mish"

    @staticmethod
    def required_keys():
        return []


class Softmax(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.layer = nn.Softmax(dim=kwargs["dim"])

    @staticmethod
    def default_values():
        return {}

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def node_type():
        return "softmax"

    @staticmethod
    def required_keys():
        return ["dim"]


ACTIVATION_NODES = {
    ELU.node_type(): ELU,
    Hardshrink.node_type(): Hardshrink,
    Hardtanh.node_type(): Hardtanh,
    Hardsigmoid.node_type(): Hardsigmoid,
    LogSigmoid.node_type(): LogSigmoid,
    ReLU.node_type(): ReLU,
    ReLU6.node_type(): ReLU6,
    LeakyReLU.node_type(): LeakyReLU,
    Hardswish.node_type(): Hardswish,
    LogSigmoid.node_type(): LogSigmoid,
    PReLU.node_type(): PReLU,
    RReLU.node_type(): RReLU,
    SELU.node_type(): SELU,
    CELU.node_type(): CELU,
    GELU.node_type(): GELU,
    Sigmoid.node_type(): Sigmoid,
    Softplus.node_type(): Softplus,
    Softshrink.node_type(): Softshrink,
    Softsign.node_type(): Softsign,
    Tanh.node_type(): Tanh,
    Tanhshrink.node_type(): Tanhshrink,
    Threshold.node_type(): Threshold,
    SiLU.node_type(): SiLU,
    Mish.node_type(): Mish,
    Identity.node_type(): Identity,
}


def build_activation_node(config: Union[dict, str]):
    if isinstance(config, str):
        return ACTIVATION_NODES[config]()

    if config["type"] not in ACTIVATION_NODES:
        raise ValueError(f"Activation type {config['type']} not supported")
    return ACTIVATION_NODES[config["type"]](config)
