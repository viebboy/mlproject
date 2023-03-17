"""
config.py: a set of experiment configurations
---------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-01
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

import itertools
import pprint
import os
from tabulate import tabulate

from mlproject.config import ConfigValue, create_all_config as _create_all_config
from mlproject.loss import MSE, MAE, CrossEntropy as CrossEntropyLoss

#TODO: put the name of your set of experiments here
# and some description
NAME = ''
DESC = ''


"""
LIST CONFIG OPTIONS

Put default or nested options below
For example, a model architecture has many parameters but we only use 1 model architecture,
we could combine all these params into a dict and pass this dict later on to ConfigValue
"""

#For example, we want to experiment with 2 types of model architecture
MODEL1_CONFIG = {}
MODEL2_CONFIG = {}

# related directories, these are ignored in git by default when creating with
# mlproject new-project
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


CACHE_SETTING = None # define your cache setting here if using async data loader
ROTATION_SETTING = None # define your rotation setting here if using async data loader


"""
TODO: MODIFY CONTENT INSIDE ALL_CONFIGS TO SPECIFY THE SET OF CONFIGURATIONS

Basically, a key in ALL_CONFIGS specifies one hyperparameter
We could list out all possible values we want to experiment with this hyperparameter by
putting them into ConfigValue()

For example, if we want to experiment with 2 values of weight decay: 1e-4 and 1e-5,
we could specify ConfigValue(1e-4, 1e-5)

ConfigValue() is essential a container that contains all possible values for a given configuration
"""

ALL_CONFIGS = {
    # number of trials to repeat a particular config
    'nb_trial': ConfigValue(5),
    # ------- model config ----------------
    'model_configs': ConfigValue(MODEL1_CONFIG, MODEL2_CONFIG), #list all possible model configs
    # -------------------------------------
    # -------- directory config -----------
    # output dir to save final results and log dir to save intermediate checkpoints
    'data_dir': ConfigValue(DATA_DIR), # this is the output dir that contains results from all exps
    'output_dir': ConfigValue(OUTPUT_DIR), # this is the output dir that contains results from all exps
    # ----------------------------------------
    # common config for data
    'nb_shard': ConfigValue(32), # sharding is used in mlproject.data.BinaryBlob
    # batch size
    'batch_size': ConfigValue(256),
    #
    # -------trainer config ----------
    # --------------------------------
    # number of epochs
    'n_epoch': ConfigValue(300),
    # loss function
    'loss_function': ConfigValue(), # define the loss function here
    # metrics:
    'metrics': ConfigValue(), # define the metrics here
    'monitor_metric': ConfigValue(), # define the monitor metric here
    'monitor_direction': ConfigValue(), # monitor direction: lower means lower is better, similar for higher
    # epoch index
    'checkpoint_idx': ConfigValue(-1), #index of the starting checkpoint, -1 it means start from the latest checkpoint
    # scheduler
    'lr_scheduler': ConfigValue('cosine'),
    # start lr
    'start_lr': ConfigValue(1e-4),
    # stop lr
    'stop_lr': ConfigValue(1e-5),
    # weight decay
    'weight_decay': ConfigValue(1e-5),
    # optimizer
    'optimizer': ConfigValue('adamW'),
    # checkpoint freq
    'checkpoint_freq': ConfigValue(100), # save checkpoint every 10 minibatch
    # max checkpoints
    'max_checkpoint': ConfigValue(-1), # save all checkpoints
    'print_freq': ConfigValue(10), # print loss value every 10 minibatch
    'eval_freq': ConfigValue(1), #  evaluate the metrics every 1 epoch
    # whether to use progress bar during evaluation
    'use_progress_bar': ConfigValue(True), # whether to use progress bar for one epoch
    # move data to device
    'move_data_to_device': ConfigValue(True),
    # retain metric objects
    'retain_metric_objects': ConfigValue(True), # if True, metric objects are saved in history, otherwise only values
    # --------- dataset server config ------------
    # if using dataset server, get_async_data_loader() will be used to retrieve
    # data loader, otherwise get_data_loader() will be used
    # for dataset_server description, take a look at
    # https://github.com/viebboy/dataset_server
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
