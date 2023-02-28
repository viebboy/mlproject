"""
entry.py: main training orchestration
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

import argparse
import os
import importlib.util
import torch
from tabulate import tabulate
from loguru import logger
from tqdm import tqdm
import time

from data import get_data_loader, get_async_data_loader, dispose_data_loader
from trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch experiment")
    parser.add_argument("--index", required=True, type=int, help="config index to run")

    # these are used to versioned default configs
    parser.add_argument(
        "--config-path",
        default='./configs/config.py',
        type=str,
        help="path to config file, should be a valid .py file. Default to config.py in the same source code"
    )

    parser.add_argument(
        "--device",
        default='cuda',
        choices=['cpu', 'cuda'],
        type=str,
        help="device to use for computation"
    )

    parser.add_argument(
        "--test-mode",
        default='False',
        type=str,
        choices=['false', 'False', 'True', 'true'],
        help="whether to run in test mode with limited data"
    )

    # process args
    args = parser.parse_args()
    if args.test_mode in ['True', 'true']:
        args.test_mode = True
    else:
        args.test_mode = False
    return args


def load_config(file: str, index: str, test_mode: bool):
    """
    load configuration values from a given config file
    a config file contains all different experiment combinations
    index refers to the index of the experiment setting to run
    """
    spec = importlib.util.spec_from_file_location('config_module', file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.get_config(index)
    config_name = config_module.get_name()
    config_description = config_module.get_description()
    config['test_mode'] = test_mode
    return config, config_name, config_description


def print_config(config_name: str, config_description: str, config: dict):
    logger.info(f'CONFIGURATION NAME: {config_name}')
    logger.info(f'CONFIGURATION DESCRIPTION: {config_description}')
    names = list(config.keys())
    table = [[name, config[name]] for name in names]
    print(tabulate(table, headers=['Config', 'Value']))


def main(
    config_file: str,
    config_name: str,
    config_description: str,
    config: dict,
    config_index: int,
    device
):
    # print config
    print_config(config_name, config_description, config)

    # -------- DATA ---------------------------------------
    if not config['use_dataset_server']:
        train_loader = get_loader(config, 'train')
        val_loader = get_loader(config, 'val')
        test_loader = get_loader(config, 'test')
    else:
        train_loader = get_async_loader(config, 'train')
        val_loader = get_async_loader(config, 'val')
        test_loader = get_async_loader(config, 'test')

    # ------------------------------------------------------

    # -------------- MODEL --------------------------------
    #TODO: create model here
    model = None
    # -----------------------------------------------------

    # create tensorboard logger here
    tensorboard_logger = None
    # prefix for this experiment
    logger_prefix = ''

    try:
        # create a trainer class
        trainer = get_trainer(
            config_file=config_file,
            config=config,
            config_name=config_name,
            config_index=config_index
        )
        trainer.fit(
            model,
            {'dataloader': train_loader},
            {'dataloader': val_loader},
            {'dataloader': test_loader},
            device=device,
            tensorboard_logger=tensorboard_logger,
            logger_prefix=logger_prefix,
        )
    except Exception as error:
        dispose_data_loader(train_loader, val_loader, test_loader)
        logger.warning('encounter the following error')
        raise error
    finally:
        # clean up
        dispose_data_loader(train_loader, val_loader, test_loader)


if __name__ == '__main__':
    args = parse_args()

    # load config file
    config_values, config_name, config_description = load_config(args.config_path, args.index, args.test_mode)
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(
        args.config_path,
        config_name,
        config_description,
        config_values,
        args.index,
        device=device,
    )
