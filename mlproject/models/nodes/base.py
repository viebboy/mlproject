"""
base.py: base node implementation
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
import torch
import torch.nn as nn
from loguru import logger


class BaseNode(nn.Module):
    def __init__(self, **kwargs: dict):
        for key in self.required_keys():
            if key not in kwargs:
                msg = "".join(
                    [
                        f"missing required parameter {key} in the configuration of the following node",
                        f'(node type: {self.node_type()}, node name: {kwargs["name"]})',
                    ]
                )
                logger.error(msg)
                raise ValueError(msg)

        super().__init__()

    def merge_with_default(self, kwargs: dict) -> dict:
        for key, value in self.default_values().items():
            if key not in kwargs:
                kwargs[key] = value

        return kwargs

    @staticmethod
    def required_keys():
        raise NotImplementedError()

    @staticmethod
    def node_type():
        raise NotImplementedError()

    @staticmethod
    def default_values():
        raise NotImplementedError()

    def initialize(self, modules=None):
        if modules is None:
            modules = self.modules()

        for layer in modules:
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)

    def infer_shape(self, *args, batch_axis: int = None):
        try:
            shape = {}
            with torch.no_grad():
                if len(args) == 1:
                    shape["in"] = list(args[0].shape)
                    if batch_axis is not None:
                        shape["in"][batch_axis] = None
                else:
                    shape["in"] = [list(arg.shape) for arg in args]
                    if batch_axis is not None:
                        for i in range(len(shape["in"])):
                            shape["in"][i][batch_axis] = None

                outputs = self.forward(*args)
                if isinstance(outputs, (tuple, list)):
                    shape["out"] = [list(output.shape) for output in outputs]
                    if batch_axis is not None:
                        for i in range(len(shape["out"])):
                            shape["out"][i][batch_axis] = None
                elif isinstance(outputs, torch.Tensor):
                    shape["out"] = list(outputs.shape)
                    if batch_axis is not None:
                        shape["out"][batch_axis] = None
                elif isinstance(outputs, dict):
                    shape["out"] = {}
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            shape["out"][key] = list(value.shape)
                            if batch_axis is not None:
                                shape["out"][key][batch_axis] = None
                        else:
                            shape["out"][key] = None
                else:
                    shape["out"] = None
        except BaseException:
            logger.warning(
                "Failed to use default implementation of infer_shape() "
                f"for node type: {self.node_type()}. "
                "You need to overwrite infer_shape() in order to support this functionality"
            )
            shape = {"in": None, "out": None}

        return shape
