"""
data.py: data handling implementation


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-02-13
* Version: 0.0.1

This is part of the cifar10 example project

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
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

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


class Dataset(CIFAR10):
    """
    Inherit from torchvision CIFAR10 dataset to provide keyword argument construction
    """
    def __init__(self, **kwargs):
        if kwargs['prefix'] == 'train':
            train = True
        else:
            train = False
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        super().__init__(root=kwargs['data_dir'], train=train, transform=transforms, download=True)

    def __getitem__(self, i: int):
        x, y = super().__getitem__(i)
        return x, torch.Tensor([y,]).long()


def dispose_data_loader(*args):
    """
    closing async loader
    """
    for item in args:
        if item is not None and isinstance(item, AsyncDataLoader):
            item.close()


def get_data_loader(config: dict, prefix: str):
    """
    Returns pytorch dataloader given the configuration and the prefix (train/validation/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/validation/test)
    Returns:
        data loader object that allows iteration
    """

    return

def get_async_data_loader(config: dict, prefix: str):
    """
    Returns async dataloader given the configuration and the prefix (train/validation/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/validation/test)
    Returns:
        mlproject.data.AsyncDataLoader object
    """

    # because our Dataset class receives 2 inputs, we put them into a
    # dictionary
    params = {
        'data_dir': config['data_dir'],
        'prefix': prefix,
    }

    # then create the loader
    loader = AsyncDataLoader(
        dataset_class=Dataset, # put the name of dataset class here
        dataset_params=params, # put the name of dataset class here,
        batch_size=config['batch_size'],
        nb_servers=config[f'{prefix}_nb_server'],
        start_port=config[f'{prefix}_start_port'],
        max_queue_size=config[f'{prefix}_max_queue_size'],
        shuffle=True if prefix == 'train' else False,
        packet_size=config['packet_size'], # size (in bytes) of packet in tcp comm
        wait_time=config['prefetch_time'], # time wait for data to be prefetched
    )

    return loader

if __name__ == '__main__':
    from tqdm import tqdm
    # test dataset
    dataset_params = {'prefix': 'train', 'data_dir': './data/'}
    dataset = Dataset(**dataset_params)
    print(f'dataset length: {len(dataset)}')
    for i in tqdm(range(len(dataset))):
        x, y = dataset[i]
        pass
    print('complete looping through dataset')

    # test dataloader
    config = {
        'data_dir': './data/',
        'batch_size': 64,
        'train_nb_server': 1,
        'train_start_port': 11111,
        'train_max_queue_size': 10,
        'packet_size': 125000,
        'prefetch_time': 5,
    }
    loader = get_async_data_loader(config, 'train')
    print(f'number of minibatch: {len(loader)}')
    for x, y in loader:
        print(f'x type: {type(x)}, x shape: {x.shape}')
        print(f'y type: {type(y)}, y shape: {y.shape}')
        pass
    loader.close()
    print('complete testing async loader')
