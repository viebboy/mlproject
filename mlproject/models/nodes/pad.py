"""
pad.py: padding node implementation
-----------------------------------


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


class ConstantPad1d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ConstantPad1d(kwargs["padding"], kwargs["value"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "constant_pad_1d"

    @staticmethod
    def default_values():
        return {"padding": 0, "value": 0}

    @staticmethod
    def required_keys():
        return []


class ConstantPad2d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ConstantPad2d(kwargs["padding"], kwargs["value"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "constant_pad_2d"

    @staticmethod
    def default_values():
        return {"padding": 0, "value": 0}

    @staticmethod
    def required_keys():
        return []


class ConstantPad3d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ConstantPad3d(kwargs["padding"], kwargs["value"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "constant_pad_3d"

    @staticmethod
    def default_values():
        return {"padding": 0, "value": 0}

    @staticmethod
    def required_keys():
        return []


class ReflectionPad1d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReflectionPad1d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "reflection_pad_1d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


class ReflectionPad2d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReflectionPad2d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "reflection_pad_2d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


class ReflectionPad3d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReflectionPad3d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "reflection_pad_3d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


class ReplicationPad1d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReplicationPad1d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "replication_pad_1d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


class ReplicationPad2d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReplicationPad2d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "replication_pad_2d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


class ReplicationPad3d(BaseNode):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.padding = nn.ReplicationPad3d(kwargs["padding"])

    def forward(self, x):
        return self.padding(x)

    @staticmethod
    def node_type():
        return "replication_pad_3d"

    @staticmethod
    def default_values():
        return {"padding": 0}

    @staticmethod
    def required_keys():
        return []


PAD_NODES = {
    ConstantPad1d.node_type(): ConstantPad1d,
    ConstantPad2d.node_type(): ConstantPad2d,
    ConstantPad3d.node_type(): ConstantPad3d,
    ReflectionPad1d.node_type(): ReflectionPad1d,
    ReflectionPad2d.node_type(): ReflectionPad2d,
    ReflectionPad3d.node_type(): ReflectionPad3d,
    ReplicationPad1d.node_type(): ReplicationPad1d,
    ReplicationPad2d.node_type(): ReplicationPad2d,
    ReplicationPad3d.node_type(): ReplicationPad3d,
}
