"""
data.py: data handling implementation
-------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-12-20
* Version: 0.0.1

This is part of the cifar10_distributed project

License
-------
Apache 2.0 License

"""

from __future__ import annotations
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from swift_loader import SwiftLoader


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
    raise NotImplementedError()


def get_swift_loader(config: dict, prefix: str):
    """
    Returns SwiftDataLoader given the configuration and the prefix (train/val/test)
    Args:
        config (dict): a dictionary that contains all configuration needed to construct the dataloader
        prefix (str): a string that indicates the type of dataset (train/val/test)

    Returns:
        swift_loader.SwiftDataLoader

    """
    from swift_loader import SwiftDataLoader

    dataset_kwargs = {
        "data_dir": config["data_dir"],
        "prefix": prefix,
    }
    dataset_class = Dataset

    # then create the loader
    loader = SwiftDataLoader(
        dataset_class=dataset_class,
        dataset_kwargs=dataset_kwargs,
        batch_size=config["batch_size"],
        shuffle=True,
        nb_consumer=config["nb_consumer"],
        worker_per_consumer=config["worker_per_consumer"],
    )

    return loader


if __name__ == "__main__":
    pass
