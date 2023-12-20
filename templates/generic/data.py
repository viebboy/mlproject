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


def dispose_data_loader(*args):
    """
    closing swift loader
    """
    for item in args:
        if item is not None and hasattr(item, "close") and callable(item.close):
            item.close()


def get_torch_loader(config: dict, prefix: str):
    """
    Returns pytorch dataloader given the configuration and the prefix (train/val/test)
    Parameters:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/val/test)
    Returns:
        data loader object that allows iteration
    """

    return


def get_swift_loader(config: dict, prefix: str):
    """
    Returns SwiftDataLoader given the configuration and the prefix (train/val/test)
    Args:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/val/test)

    Returns:
        swift_loader.SwiftDataLoader

    """
    from swift_loader import SwiftLoader

    # TODO: construct a dictionary of dataset parameters here
    dataset_kwargs = {}
    # TODO: put the name of dataset class here
    dataset_class = None

    # then create the loader
    loader = SwiftLoader(
        dataset_class=dataset_class,
        dataset_kwargs=dataset_kwargs,
        batch_size=config["batch_size"],
        shuffle=True,
        nb_consumer=config["nb_consumer"],
        worker_per_consumer=config["worker_per_consumer"],
    )

    return loader


if __name__ == "__main__":
    # TODO: test your custom dataset and loader here
    pass
