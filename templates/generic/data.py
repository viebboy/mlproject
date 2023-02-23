"""
data.py: data handling implementation
-------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-01
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

from __future__ import annotations
import numpy as np
import os
from loguru import logger
import joblib
import dill
import json
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from cvinfer.common import (
    Frame,
    BoundingBox,
    Point,
    OnnxModel,
)
from mlproject.data import (
    BinaryBlob,
    CacheDataset,
    ConcatDataset,
)
from dataset_server import AsyncDataLoader


def dispose_data_loader(*args):
    """
    closing async loader
    """
    for item in args:
        if item is not None and isinstance(item, AsyncDataLoader):
            item.close()


def get_data_loader(config: dict, prefix: str):
    """
    Returns pytorch dataloader given the configuration and the prefix (train/val/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/val/test)
    Returns:
        data loader object that allows iteration
    """

    return

def get_async_data_loader(config: dict, prefix: str):
    """
    Returns async dataloader given the configuration and the prefix (train/val/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/val/test)
    Returns:
        mlproject.data.AsyncDataLoader object
    """

    # TODO: construct a dictionary of dataset parameters here
    dataset_params = {
    }
    # TODO: put the name of dataset class here
    dataset_class = None

    # then create the loader
    loader = AsyncDataLoader(
        dataset_class=dataset_class,
        dataset_params=dataset_params,
        batch_size=config['batch_size'],
        nb_servers=config[f'{prefix}_nb_server'],
        start_port=config[f'{prefix}_start_port'],
        max_queue_size=config[f'{prefix}_max_queue_size'],
        shuffle=True if prefix == 'train' else False,
        packet_size=config['packet_size'], # size (in bytes) of packet in tcp comm
    )

    return loader

if __name__ == '__main__':
    #TODO: test your custom dataset and loader here
