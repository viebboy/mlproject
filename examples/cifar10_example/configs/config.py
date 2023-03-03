"""
config.py: a set of experiment configurations


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-02-13
* Version: 0.0.1

This is part of the cifar10 example project

License
-------
Apache 2.0 License

"""

import itertools
import pprint
import os

from mlproject.config import ConfigValue, create_all_config as _create_all_config
from mlproject.loss import MSE, MAE, CrossEntropy as CrossEntropyLoss
from mlproject.metric import (
    Accuracy,
    CrossEntropy,
    F1,
    Precision,
    Recall,
)
import torch.nn as nn

NAME = 'cifar10 experiments with densenet'
DESC = 'cifar10 experiments with densenet'


"""
LIST CONFIG OPTIONS
"""

# related directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

CACHE_SETTING = None
ROTATION_SETTING = None

#For example, we want to experiment with 2 types of model architecture
# model1: with global averaging
MODEL1_CONFIG = {
    'nb_init_filters': 24,
    'growth_rates': [(16, 16), (16, 16)],
    'bottleneck': 0.5,
    'bottleneck_style': 'in',
    'activation': nn.SiLU(inplace=True),
    'groups': 1,
    'reduce_input': False,
    'dropout': None,
    'pool_in_last_block': True,
    'global_average': True,
    'use_bias_for_embedding_layer': False,
    'embedding_dim': 128,
    'input_height': 32,
    'input_width': 32,
    'use_bias_for_output_layer': True,
    'nb_class': 10,
}

# model2 without global averaging
MODEL2_CONFIG = {
    'nb_init_filters': 24,
    'growth_rates': [(16, 16), (16, 16)],
    'bottleneck': 0.5,
    'bottleneck_style': 'in',
    'activation': nn.SiLU(inplace=True),
    'groups': 1,
    'reduce_input': False,
    'dropout': None,
    'pool_in_last_block': True,
    'global_average': False,
    'use_bias_for_embedding_layer': False,
    'embedding_dim': 128,
    'input_height': 32,
    'input_width': 32,
    'use_bias_for_output_layer': True,
    'nb_class': 10,
}


"""
TODO: MODIFY CONTENT INSIDE ALL_CONFIGS TO SPECIFY THE SET OF CONFIGURATIONS
"""


ALL_CONFIGS = {
    # model config
    'model_config': ConfigValue(MODEL1_CONFIG, MODEL2_CONFIG), #list all possible model configs
    # output dir to save final results and log dir to save intermediate checkpoints
    'data_dir': ConfigValue(DATA_DIR), # this is the data dir that contains cifar10 data (will be downloaded)
    'output_dir': ConfigValue(OUTPUT_DIR), # this is the output dir that contains results from all exps
    #
    # common config for data
    'nb_shard': ConfigValue(8),
    # batch size
    'batch_size': ConfigValue(4),
    #
    # -------trainer config ----------
    # --------------------------------
    # number of epochs
    'n_epoch': ConfigValue(10),
    # loss function
    'loss_function': ConfigValue(CrossEntropyLoss), # define the loss function here
    # metrics:
    # remember: metrics is a list of metric objects
    'metrics': ConfigValue([CrossEntropy(), Accuracy(), Precision(), Recall(), F1()]),
    'monitor_metric': ConfigValue(Accuracy().name()), # define the name of monitor metric here
    'monitor_direction': ConfigValue('higher'), # monitor direction: lower means lower is better, similar for higher
    # epoch index
    'checkpoint_idx': ConfigValue(-1), #index of the checkpoint to load. -1 means the latest checkpoint
    # scheduler
    'lr_scheduler': ConfigValue('cosine'),
    # start lr
    'start_lr': ConfigValue(1e-4),
    # stop lr
    'stop_lr': ConfigValue(1e-5),
    # weight decay
    'weight_decay': ConfigValue(1e-4),
    # optimizer
    'optimizer': ConfigValue('adamW'),
    # checkpoint freq
    'checkpoint_freq': ConfigValue(1000), # save checkpoint every 20 minibatch
    'print_freq': ConfigValue(200), # print loss value for every 10 minibatch
    'eval_freq': ConfigValue(1), # evaluate the metrics for every 1 epoch
    # max checkpoints
    'max_checkpoint': ConfigValue(10), # only retain the last 10 checkpoints
    # whether to use progress bar
    'use_progress_bar': ConfigValue(True), # whether to use progress bar in evaluation
    # move data to device
    'move_data_to_device': ConfigValue(True), # whether data generated from dataloader requires moving to device
    # retain metric objects
    'retain_metric_objects': ConfigValue(True), # if False, save only the metric values (not obj) in history
    # --------- dataset server config ------------
    # --------------------------------------------
    'use_async_loader': ConfigValue(True), # whether to use dataset_server or Torch's dataloader
    # options for train data
    'train_nb_worker': ConfigValue(1),
    'train_max_queue_size': ConfigValue(100),
    # options for val data
    'val_nb_worker': ConfigValue(1),
    'val_max_queue_size': ConfigValue(100),
    # options for test data
    'test_nb_worker': ConfigValue(1),
    'test_max_queue_size': ConfigValue(100),
    'cache_setting': ConfigValue(CACHE_SETTING),
    'rotation_setting': ConfigValue(ROTATION_SETTING),
    'nearby_shuffle': ConfigValue(100), # nearby shuffling within 100 samples
    'use_threading_in_data_loader': ConfigValue(False), # whether to use another thread to communicate with other proc
}


"""
DO NOT MODIFY CONTENT BELOW
"""


def create_all_config():
    all_configs = _create_all_config(ALL_CONFIGS)
    return all_configs


def print_all_config():
    configs = create_all_config()
    for idx, item in enumerate(configs):
        msg = f'config_index={idx}'
        print('-' * len(msg))
        print(msg)
        print('-' * len(msg))
        pprint.pprint(item)


def get_config(index):
    # compute all configurations first
    configs = create_all_config()
    if index < len(configs):
        return configs[index]
    else:
        print(f'number of configurations: {len(configs)}')
        print(f'configuration index {index}')
        raise RuntimeError('the given index is out of range from all configurations')

def get_name():
    if NAME == '':
        raise RuntimeError('Name of the configuration must not be empty')
    return NAME

def get_description():
    if DESC == '':
        raise RuntimeError('Description of the configuration must not be empty')
    return DESC
