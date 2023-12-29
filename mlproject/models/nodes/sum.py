"""
sum.py: sum node
----------------


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
from mlproject.models.nodes.base import BaseNode


class Sum(BaseNode):
    """
    Summation node
    """

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        # get default values
        kwargs = self.merge_with_default(kwargs)

    def forward(self, *inputs):
        x = 0
        for item in inputs:
            x += item
        return x

    @staticmethod
    def required_keys():
        return []

    @staticmethod
    def node_type():
        return "Sum"

    @staticmethod
    def default_values():
        return {}
