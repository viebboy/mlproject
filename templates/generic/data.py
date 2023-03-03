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
from dataset_server import DataLoader


def dispose_data_loader(*args):
    """
    closing async loader
    """
    for item in args:
        if item is not None and isinstance(item, DataLoader):
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
        mlproject.data.DataLoader object
    """

    # TODO: construct a dictionary of dataset parameters here
    dataset_params = {
    }
    # TODO: put the name of dataset class here
    dataset_class = None

    if config['cache_setting'] is not None:
        cache_setting = {key: value for key, value in config['cache_setting'].items()}
        # append the dataset prefix
        cache_setting['prefix'] = cache_setting['prefix'] + f'_{prefix}'
    else:
        cache_setting = None

    if config['rotation_setting'] is not None:
        rotation_setting = {key: value for key, value in config['rotation_setting'].items()}
        # append the dataset prefix
        if rotation_setting['medium'] == 'disk':
            rotation_setting['prefix'] = rotation_setting['prefix'] + f'_{prefix}'
    else:
        rotation_setting = None

    # then create the loader
    loader = DataLoader(
        dataset_class=dataset_class,
        dataset_params=dataset_params,
        batch_size=config['batch_size'],
        nb_worker=config[f'{prefix}_nb_worker'],
        max_queue_size=config[f'{prefix}_max_queue_size'],
        shuffle=True if prefix == 'train' else False,
        nearby_shuffle=config['nearby_shuffle'],
        cache_setting=cache_setting,
        rotation_setting=rotation_setting,
        use_threading=config['use_threading_in_data_loader'],
    )

    return loader

if __name__ == '__main__':
    #TODO: test your custom dataset and loader here
    pass
