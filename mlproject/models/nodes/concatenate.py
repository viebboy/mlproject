"""
concatenate.py: concatenate node
--------------------------------


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
from mlproject.models.nodes.base import BaseNode


class Concatenate(BaseNode):
    """
    Concatenation node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)
        self.axis = kwargs["axis"]

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.axis)

    @staticmethod
    def required_keys():
        return ["axis"]

    @staticmethod
    def node_type():
        return "Concatenate"

    @staticmethod
    def default_values():
        return {}
