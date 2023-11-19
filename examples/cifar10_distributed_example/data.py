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
from dataset_server import DataLoader


class Dataset(CIFAR10):
    """
    Inherit from torchvision CIFAR10 dataset to provide keyword argument construction
    """

    def __init__(self, **kwargs):
        if kwargs["prefix"] == "train":
            train = True
        else:
            train = False
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        super().__init__(
            root=kwargs["data_dir"], train=train, transform=transforms, download=True
        )

    def __getitem__(self, i: int):
        x, y = super().__getitem__(i)
        return (
            x,
            torch.Tensor(
                [
                    y,
                ]
            ).long(),
        )


def dispose_data_loader(*args):
    """
    closing async loader
    """
    for item in args:
        if item is not None and isinstance(item, DataLoader):
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
    params = {
        "data_dir": config["data_dir"],
        "prefix": prefix,
    }
    dataset = Dataset(**params)
    data_loader = TorchDataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True if prefix == "train" else False,
    )

    return data_loader


def get_async_data_loader(config: dict, prefix: str):
    """
    Returns async dataloader given the configuration and the prefix (train/validation/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/validation/test)
    Returns:
        mlproject.data.DataLoader object
    """

    # because our Dataset class receives 2 inputs, we put them into a
    # dictionary
    params = {
        "data_dir": config["data_dir"],
        "prefix": prefix,
    }

    if config["cache_setting"] is not None:
        cache_setting = {key: value for key, value in config["cache_setting"].items()}
        # append the dataset prefix
        cache_setting["prefix"] = cache_setting["prefix"] + f"_{prefix}"
    else:
        cache_setting = None

    if config["rotation_setting"] is not None:
        rotation_setting = {
            key: value for key, value in config["rotation_setting"].items()
        }
        # append the dataset prefix
        if rotation_setting["medium"] == "disk":
            rotation_setting["prefix"] = rotation_setting["prefix"] + f"_{prefix}"
    else:
        rotation_setting = None

    # then create the loader
    loader = DataLoader(
        dataset_class=Dataset,
        dataset_params=params,
        batch_size=config["batch_size"],
        nb_worker=config[f"{prefix}_nb_worker"],
        max_queue_size=config[f"{prefix}_max_queue_size"],
        shuffle=True if prefix == "train" else False,
        nearby_shuffle=config["nearby_shuffle"],
        cache_setting=cache_setting,
        rotation_setting=rotation_setting,
        use_threading=config["use_threading_in_data_loader"],
    )

    return loader


if __name__ == "__main__":
    from tqdm import tqdm

    # test dataset
    dataset_params = {"prefix": "train", "data_dir": "./data/"}
    dataset = Dataset(**dataset_params)
    print(f"dataset length: {len(dataset)}")
    for i in tqdm(range(len(dataset))):
        x, y = dataset[i]
        pass
    print("complete looping through dataset")

    # test dataloader
    config = {
        "data_dir": "./data/",
        "batch_size": 64,
        "train_nb_server": 1,
        "train_start_port": 11111,
        "train_max_queue_size": 10,
        "packet_size": 125000,
        "prefetch_time": 5,
    }
    loader = get_async_data_loader(config, "train")
    print(f"number of minibatch: {len(loader)}")
    for x, y in loader:
        print(f"x type: {type(x)}, x shape: {x.shape}")
        print(f"y type: {type(y)}, y shape: {y.shape}")
        pass
    loader.close()
    print("complete testing async loader")
