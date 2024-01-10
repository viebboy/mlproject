"""
__init__.py: init for node module
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
from loguru import logger

from mlproject.models.nodes.base import BaseNode
from mlproject.models.nodes.activations import ACTIVATION_NODES
from mlproject.models.nodes.pool import POOL_NODES
from mlproject.models.nodes.pad import PAD_NODES
from mlproject.models.nodes.concatenate import Concatenate
from mlproject.models.nodes.sum import Sum
from mlproject.models.nodes.conv1d import Conv1D
from mlproject.models.nodes.conv2d import Conv2D
from mlproject.models.nodes.conv_bn_act_1d import ConvBnAct1D
from mlproject.models.nodes.conv_bn_act_2d import ConvBnAct2D
from mlproject.models.nodes.batchnorm1d import BatchNorm1D
from mlproject.models.nodes.batchnorm2d import BatchNorm2D
from mlproject.models.nodes.bilinear_input_normalize import BilinearInputNormalize
from mlproject.models.nodes.bilinear_transform import BilinearTransform
from mlproject.models.nodes.tabl import TABL


_NODES = {
    **ACTIVATION_NODES,
    **POOL_NODES,
    **PAD_NODES,
    Concatenate.node_type(): Concatenate,
    Sum.node_type(): Sum,
    Conv1D.node_type(): Conv1D,
    Conv2D.node_type(): Conv2D,
    ConvBnAct1D.node_type(): ConvBnAct1D,
    ConvBnAct2D.node_type(): ConvBnAct2D,
    BatchNorm1D.node_type(): BatchNorm1D,
    BatchNorm2D.node_type(): BatchNorm2D,
    BilinearInputNormalize.node_type(): BilinearInputNormalize,
    BilinearTransform.node_type(): BilinearTransform,
    TABL.node_type(): TABL,
}


def register_node(node_builder: callable) -> None:
    if not issubclass(node_builder, BaseNode):
        logger.error("A valid node must be a subclass of BaseNode")
        raise ValueError("A valid node must be a subclass of BaseNode")

    node_type = node_builder.node_type()
    if node_type in _NODES:
        raise ValueError(f"Node type {node_type} already exists")

    _NODES[node_type] = node_builder

    return node_builder


def build_node(node_config: dict) -> BaseNode:
    if node_config["type"] not in _NODES:
        raise ValueError(f"Node type {node_config['type']} does not exist")
    try:
        node = _NODES[node_config["type"]](**node_config)
    except Exception as error:
        logger.error(f'failed to build node {node_config["name"]}')
        logger.error(f"faced error: {error}")
        raise error

    return node


def get_all_node_types():
    return list(_NODES.keys())
