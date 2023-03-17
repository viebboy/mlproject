"""
trainer.py: custom trainer goes here
------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-01
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
from glob import glob
import pandas as pd
import plotly.express as px
import time
import numpy as np
import tempfile
from loguru import logger
import dill
from typing import Callable
import copy
import shutil

from mlproject.metric import (
    MSE,
    MAE,
    Accuracy,
    CrossEntropy,
    F1,
    Precision,
    Recall,
    MetricFromLoss,
    NestedMetric,
)
from mlproject.loss import (
    compose_losses_to_callable,
    get_CrossEntropy,
    get_MSE,
    get_MAE,
)
from mlproject.trainer import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler,
    Trainer as BaseTrainer,
)
from mlproject.config import get_config_string


class Trainer(BaseTrainer):
    """
    Custom trainer class. Subclass BaseTrainer and overwrite necessary methods
    For customization, one should
    - overwrite and reimplement update_loop() to change the training logic of one epoch if needed
    - overwrite and reimplement eval() to change the evaluation logics if needed
    - overwrite and reimplement export_to_onnx() if the model is a multi-input or multi-output

    The original logics is included in the template for reference
    There are certain parts one should not change to ensure the trainer works properly
    """
    def __init__(
        self,
        n_epoch: int,
        output_dir: str,
        loss_function: Callable,
        metrics: list,
        monitor_metric: str,
        monitor_direction: str,
        checkpoint_idx: int=-1,
        lr_scheduler: Callable=get_cosine_lr_scheduler(1e-3, 1e-5),
        optimizer: str='adam',
        weight_decay: float=1e-4,
        log_dir: str=None,
        checkpoint_freq: int=10,
        max_checkpoint: int=10,
        eval_freq: int=1,
        print_freq: int=10,
        use_progress_bar: bool=True,
        test_mode: bool=False,
        move_data_to_device: bool=True,
        retain_metric_objects: bool=True,
        sample_input=None,
        logger=logger,
    ):
        super().__init__(
            n_epoch=n_epoch,
            output_dir=output_dir,
            loss_function=loss_function,
            metrics=metrics,
            monitor_metric=monitor_metric,
            monitor_direction=monitor_direction,
            checkpoint_idx=checkpoint_idx,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            weight_decay=weight_decay,
            log_dir=log_dir,
            checkpoint_freq=checkpoint_freq,
            max_checkpoint=max_checkpoint,
            eval_freq=eval_freq,
            print_freq=print_freq,
            use_progress_bar=use_progress_bar,
            test_mode=test_mode,
            move_data_to_device=move_data_to_device,
            retain_metric_objects=retain_metric_objects,
            sample_input=sample_input,
            logger=logger,
        )

    def update_loop(self, model, data, optimizer, device):
        """
        The update loop of one epoch
        One should only change the logic of model prediction and loss computation if needed
        """
        # counter for profiling
        total_time = 0
        forward_time = 0
        backward_time = 0
        data_prep_time = 0
        data_to_gpu_time = 0
        profile_counter = 0

        # always remember to put the model in training mode
        model.train()

        start_stamp = time.time()

        for inputs, labels in data['dataloader']:
            pre_gpu_stamp = time.time()

            if self.move_data_to_device:
                # move data to device, this works even with nested list
                inputs = self._move_data_to_device(inputs, device)
                labels = self._move_data_to_device(labels, device)

            pre_forward_stamp = time.time()
            predictions = model(inputs)
            post_forward_stamp = time.time()

            # this is the place where one might need to reimplement
            loss = self.loss_function(predictions, labels)

            # calling backward and update weights
            loss.backward()
            optimizer.step()
            backward_stamp = time.time()
            optimizer.zero_grad()

            # ------- Do Not Modify -----------------------------------
            # the logic after this is necessary for the trainer to works
            # ----------------------------------------------------------
            # accumulate loss function
            self.accumulated_loss += loss.item()
            self.accumulated_loss_counter += 1

            # update information for profiling
            forward_time += (post_forward_stamp - pre_forward_stamp)
            backward_time += (backward_stamp - post_forward_stamp)
            total_time += (backward_stamp - start_stamp)
            data_prep_time += (pre_gpu_stamp - start_stamp)
            data_to_gpu_time += (pre_forward_stamp - pre_gpu_stamp)
            profile_counter += 1

            # increment counter for minibatch
            # minibatch_idx is index in the current epoch
            # cur_minibatch is index in the whole training loop
            self.minibatch_idx += 1
            self.cur_minibatch += 1

            # record the sample input for ONNX export if user doesnt provide
            if self.sample_input is None:
                self.sample_input = self._move_data_to_device(inputs, torch.device('cpu'))

            if (self.cur_minibatch % self.print_freq) == 0:
                # printing if needed
                self.print_and_update(
                    total_time,
                    data_prep_time,
                    data_to_gpu_time,
                    forward_time,
                    backward_time,
                    profile_counter,
                )
                # reset
                total_time = 0
                data_prep_time = 0
                data_to_gpu_time = 0
                forward_time = 0
                backward_time = 0
                profile_counter = 0

            epoch_ended = self.minibatch_idx == self.total_minibatch
            if (self.cur_minibatch % self.checkpoint_freq) == 0:
                # checkpoint every K minibatch
                self.update_checkpoint(model, optimizer, epoch_ended)
                # move back to device because exporting a model will move it
                # back to cpu and put it in eval mode
                model.to(device)
                model.train()

            start_stamp = time.time()

            if self.test_mode and self.minibatch_idx > self.total_minibatch:
                # early stopping for test mode
                break

        # reset index within an epoch
        self.minibatch_idx = 0

    def eval(self, model: torch.nn.Module, data: dict, device, dataset_name: str):
        """
        Evaluation logics
        If data is None, empty dict should be returned
        """

        if data is None or data['dataloader'] is None:
            return {}

        self.logger.info(f'evaluating {dataset_name}...')

        # always remember to put model in eval mode
        model.eval()

        # reset the metric objects
        # this is important, metrics are instances of metric objects
        # every time we need to re-evaluate, we need to reset this object
        # to flush previous statistics
        for m in self.metrics:
            m.reset()

        # create fresh copy of metric objects
        # this is because the metric objects are saved in history
        # we don't want to append the same reference for every evaluation
        metrics = copy.deepcopy(self.metrics)

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(data['dataloader']))
        else:
            total_minibatch = len(data['dataloader'])

        if self.use_progress_bar:
            loader = tqdm(data['dataloader'], desc=f'#Evaluating {dataset_name}: ', ncols=120, ascii=True)
        else:
            loader = data['dataloader']

        with torch.no_grad():
            for minibatch_idx, (inputs, labels) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                if self.move_data_to_device:
                    inputs = self._move_data_to_device(inputs, device)
                    labels = self._move_data_to_device(labels, device)

                predictions = model(inputs)
                for m in metrics:
                    m.update(predictions=predictions, labels=labels)

        # note here that we are collecting the metric object
        # not just the value
        performance = {}
        if self.retain_metric_objects:
            for m in metrics:
                performance[m.name()] = m
        else:
            for m in metrics:
                performance[m.name()] = m.value()

        return performance

    def export_to_onnx(self, model, sample_input, onnx_path):
        if sample_input is None:
            raise RuntimeError('sample_input is None; exporting a model to ONNX requires sample input')

        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={
                'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}
            }
        )
        self.logger.info(f'save model in ONNX format in {onnx_path}')


def get_trainer(config_file: str, config: dict, config_name: str, config_index: int):
    """
    Returns trainer object
    parameters:
        config_file (str): path to the file that holds all configs
        config (dict): a dictionary that holds a particular config for this experiment
        config_name (str): name of the config
        config_index (int): index of the config (out of all configurations specified in config_file)

    returns:
        a trainer instance
    """

    # prep output dir and log dir
    # note this is the metadir for all experiments
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])

    # need to create subdir as output dir and log dir for this particular exp
    config_name = config_name.replace(' ', '_')
    # output directory path has the following structure:
    # user_given_dir/config_name/config_index/trial_index/
    output_dir = os.path.join(
        config['output_dir'],
        config_name,
        '{:09d}'.format(config_index),
        '{:09d}'.format(config['trial_index']),
    )
    # under the output dir is another subdir for checkpoints
    log_dir = os.path.join(output_dir, 'checkpoints')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # now copy the original config file, the config index and the content of
    # the config to log and output dir for reference purposes
    shutil.copy(config_file, os.path.join(log_dir, os.path.basename(config_file)))
    shutil.copy(config_file, os.path.join(output_dir, os.path.basename(config_file)))
    with open(os.path.join(log_dir, f'config_index_{config_index}.dill'), 'wb') as fid:
        dill.dump(config, fid)
    with open(os.path.join(output_dir, f'config_index_{config_index}.dill'), 'wb') as fid:
        dill.dump(config, fid)

    # also save the textual content of the config in a text file
    with open(os.path.join(log_dir, f'config_content.txt'), 'w') as fid:
        fid.write(get_config_string(config, config_index))

    with open(os.path.join(output_dir, f'config_content.txt'), 'w') as fid:
        fid.write(get_config_string(config, config_index))
    # ---------------------------------------------------------

    # create lr scheduler
    if config['lr_scheduler'] == 'cosine':
        lr_scheduler = get_cosine_lr_scheduler(config['start_lr'], config['stop_lr'])
    elif config['lr_scheduler'] == 'multiplicative':
        lr_scheduler = get_multiplicative_lr_scheduler(
            config['start_lr'],
            config['epochs_to_drop_lr'], #the epoch number to drop learning rate
            config['lr_multiplicative_factor'], #factor to multiply to the learning rate to change
        )
    else:
        raise NotImplemented

    # parse loss function here

    # TODO: create a trainer here
    trainer =  Trainer(
        n_epoch=config['n_epoch'],
        output_dir=output_dir, #use directory created above
        loss_function=config['loss_function'],
        metrics=config['metrics'],
        monitor_metric=config['monitor_metric'],
        monitor_direction=config['monitor_direction'],
        checkpoint_idx=config['checkpoint_idx'],
        lr_scheduler=lr_scheduler,
        optimizer=config['optimizer'],
        weight_decay=config['weight_decay'],
        log_dir=log_dir, # use directory created above for this configuration
        checkpoint_freq=config['checkpoint_freq'],
        max_checkpoint=config['max_checkpoint'],
        eval_freq=config['eval_freq'],
        print_freq=config['print_freq'],
        use_progress_bar=config['use_progress_bar'],
        test_mode=config['test_mode'],
        move_data_to_device=config['move_data_to_device'],
        retain_metric_objects=config['retain_metric_objects'],
    )
    return trainer
