"""
config.py: a set of experiment configurations
---------------------------------------------


* Copyright: Dat Tran (viebboy@gmail.com)
* Authors: Dat Tran
* Date: #TODO
* Version: 0.0.1

This is part of the #TODO project

License
-------
Proprietary License

"""

import os
from mlproject.config import ConfigValue

# absolute path to the directory containing this file when running
# this provides a good reference if you want join path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# and some description
NAME = "fill_your_exp_name"
DESC = "fill_your_description"

# LOSS SETUP
LOSS = {
    # path to .py implementation
    # it should contain get_loss_function(**arguments) method that returns a callable
    # the returned callable should have signature: loss(predictions, labels) using torch
    "implementation": os.path.join(CURRENT_DIR, "loss.py"),
    # arguments should always be a dict, can be empty
    "arguments": {},
}

METRIC = {
    # path to .py implementation
    # it should contain get_metric(**arguments) method that returns a dictionary
    # the dict should contain
    # - metrics (list): a list of mlproject.metrics.Metric objects
    # - monitor_metric (str): name of the metric that is used to monitor best weights
    # - monitor_direction (str): can be `higher` or `lower`, indicating the direction of the monitor_metric
    "implementation": os.path.join(CURRENT_DIR, "metric.py"),
    # arguments should always be a dict, can be empty
    "arguments": {},
}

DATASET = {
    "implementation": os.path.join(CURRENT_DIR, "dataset.py"),
    "train_arguments": {},
    "val_arguments": {},
    "test_arguments": {},
}


# these are the settings for SwiftLoader
# https://github.com/viebboy/swift-loader
DATA_LOADER = {
    "batch_size": 32,
    # amount of CPUs consumed by each worker (GPU) to load data
    # if using CPU, then we only have 1 consumer (of the data)
    # if using GPU, the the number of consumers is the number of GPUs
    "worker_per_consumer": 4,
}

MODEL = {
    "implementation": os.path.join(CURRENT_DIR, "model.py"),
    "arguments": {},
}

ALL_CONFIGS = {
    "output_dir": ConfigValue("fill_path_to_output_dir_that_contains_all_results"),
    "n_epoch": ConfigValue(20),
    # dataset config
    "dataset": ConfigValue(DATASET),
    # dataloader
    "dataloader": ConfigValue(DATA_LOADER),
    # model config
    "model": ConfigValue(MODEL),
    # loss config
    "loss": ConfigValue(LOSS),  # define different loss setup here
    # metric config
    "metric": ConfigValue(METRIC),
    # epoch index
    "checkpoint_idx": ConfigValue(
        -1
    ),  # index of the starting checkpoint, -1 it means start from the latest checkpoint
    # scheduler
    "lr_scheduler": ConfigValue("cosine"),
    # start lr
    "start_lr": ConfigValue(1e-3),
    # stop lr
    "stop_lr": ConfigValue(1e-5),
    # weight decay
    "weight_decay": ConfigValue(1e-5),
    # grad accumulation, 1 means no accumulate over multiple steps
    "grad_accumulation_step": ConfigValue(10),
    # optimizer
    "optimizer": ConfigValue("adamW"),
    # checkpoint freq: save every K minibatch
    # if None: save checkpoint every epoch
    # if value K is given, save every K minibatch
    "checkpoint_freq": ConfigValue(None),
    # max checkpoints: -1 means keeping all checkpoints
    "max_checkpoint": ConfigValue(5),
    # load best: whether to load the best checkpoint based on train or val (if exists)
    # monitor_metric
    "load_best": ConfigValue(True),
    # print frequency: minibatch interval to print loss
    "print_freq": ConfigValue(20),  # print loss value every 20 minibatch
    # synchronized_print: if True, only print from main process
    "synchronized_print": ConfigValue(True),
    # epoch interval to perform eval
    "eval_freq": ConfigValue(1),  # eval every 1 epoch
    # whether to use progress bar during evaluation
    "use_progress_bar": ConfigValue(True),  # whether to use progress bar for one epoch
    # whether to save metric as values (False) or save as objects (True)
    "retain_metric_objects": ConfigValue(
        False
    ),  # if True, metric objects are saved in history, otherwise only values
}
