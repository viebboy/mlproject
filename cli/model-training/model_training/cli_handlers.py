"""
cli_handlers.py: handlers for each cli sub-command
--------------------------------------------------


* Copyright: 2022 datsbot.com
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-10-28
* Version: 0.0.1


This is part of ml package

License
-------
Proprietary License

"""

from __future__ import annotations
import argparse
import os
import torch
import shutil
from model_training.train_entry import main as train_entry_main
from model_training.tools import load_config, get_config_size


def write_template_handler(args: argparse.Namespace) -> None:
    os.makedirs(args.path, exist_ok=True)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(current_dir, "template")
    config_file = os.path.join(template_dir, "config.py")
    dataset_file = os.path.join(template_dir, "dataset.py")
    loss_file = os.path.join(template_dir, "loss.py")
    metric_file = os.path.join(template_dir, "metric.py")

    shutil.copy(config_file, os.path.join(args.path, "config.py"))
    shutil.copy(dataset_file, os.path.join(args.path, "dataset.py"))
    shutil.copy(loss_file, os.path.join(args.path, "loss.py"))
    shutil.copy(metric_file, os.path.join(args.path, "metric.py"))


def train_single_config_handler(args: argparse.Namespace) -> None:
    """sample handler for a given sub-command"""

    if "MLPROJECT_DEVICE_COUNT" in os.environ:
        nb_consumer = int(os.environ["MLPROJECT_DEVICE_COUNT"])
    else:
        nb_consumer = max(1, torch.cuda.device_count())
        os.environ["MLPROJECT_DEVICE_COUNT"] = str(nb_consumer)
        if nb_consumer == 1 and torch.cuda.is_available():
            print("RUNNING WITH 1 GPU")
        elif nb_consumer > 1:
            print(f"RUNNING WITH {nb_consumer} GPUs")

    config_values = load_config(args.config_path, args.index, args.test_mode)
    train_entry_main(
        config_values,
        device=args.device,
        nb_consumer=nb_consumer,
        cli_kwargs=args.kwargs,
    )


def train_handler(args: argparse.Namespace) -> None:
    """sample handler for a given sub-command"""

    if "MLPROJECT_DEVICE_COUNT" in os.environ:
        nb_consumer = int(os.environ["MLPROJECT_DEVICE_COUNT"])
    else:
        nb_consumer = max(1, torch.cuda.device_count())
        os.environ["MLPROJECT_DEVICE_COUNT"] = str(nb_consumer)
        if nb_consumer == 1 and torch.cuda.is_available():
            print("RUNNING WITH 1 GPU")
        elif nb_consumer > 1:
            print(f"RUNNING WITH {nb_consumer} GPUs")

    nb_config = get_config_size(args.config_path)

    for index in range(nb_config):
        raise NotImplementedError()
