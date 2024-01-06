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
from mlproject.distributed_trainer import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler,
    Trainer as BaseTrainer,
)


class Trainer(BaseTrainer):
    """
    Modify 3 methods below if needed
    """

    def update_loop(self, model, data, optimizer):
        total_time = 0
        forward_time = 0
        backward_time = 0
        data_prep_time = 0
        profile_counter = 0
        optimizer_time = 0
        optimizer_step = 0

        # important to switch to train mode
        model.train()

        start_stamp = time.perf_counter()

        epoch_ended = False
        for inputs, labels in data["dataloader"]:
            if epoch_ended:
                break

            pre_forward_stamp = time.perf_counter()
            predictions = model(inputs)
            post_forward_stamp = time.perf_counter()

            # divide loss by grad_accumulation_step
            loss = self.loss_function(predictions, labels) / self.grad_accumulation_step
            self.FABRIC.backward(loss)
            backward_stamp = time.perf_counter()

            # accumulate loss value for printing and checkpointing
            self.accumulated_loss += loss.item()
            self.accumulated_loss_counter += 1

            # update parameters when reaching accumulation step
            if (self.cur_minibatch + 1) % self.grad_accumulation_step == 0:
                pre_optimizer_stamp = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad()
                post_optimizer_stamp = time.perf_counter()
                optimizer_time += post_optimizer_stamp - pre_optimizer_stamp
                optimizer_step += 1

            # update information for profiling
            forward_time += post_forward_stamp - pre_forward_stamp
            backward_time += backward_stamp - post_forward_stamp
            total_time += backward_stamp - start_stamp
            data_prep_time += pre_forward_stamp - start_stamp
            profile_counter += 1

            # increment counter for minibatch
            # minibatch_idx is index in the current epoch
            # cur_minibatch is index in the whole training loop
            self.minibatch_idx += 1
            self.cur_minibatch += 1

            # record the sample input for ONNX export if user doesnt provide
            if self.sample_input is None:
                self.sample_input = inputs.cpu()

            if (self.cur_minibatch % self.print_freq) == 0:
                # printing if needed
                self.print_and_update(
                    total_time,
                    data_prep_time,
                    forward_time,
                    backward_time,
                    profile_counter,
                    optimizer_time,
                    optimizer_step,
                )
                # reset
                total_time = 0
                data_prep_time = 0
                forward_time = 0
                backward_time = 0
                profile_counter = 0
                optimizer_time = 0
                optimizer_step = 0

            epoch_ended = self.minibatch_idx == self.total_minibatch
            if (
                self.checkpoint_freq is not None
                and (self.cur_minibatch % self.checkpoint_freq) == 0
            ):
                # checkpoint every K minibatch
                self.update_checkpoint(model, optimizer, epoch_ended)
                # remember to change to train mode
                model.train()

            start_stamp = time.perf_counter()

            if self.test_mode and self.minibatch_idx > self.total_minibatch:
                # early stopping for test mode
                break

        # reset index within an epoch
        self.minibatch_idx = 0

    def eval(self, model: torch.nn.Module, data: dict, dataset_name: str):
        if data is None or data["dataloader"] is None:
            return {}

        self.logger.info(f"evaluating {dataset_name}...")
        model.eval()

        # reset the metric objects
        for m in self.metrics:
            m.reset()

        # create fresh copy of metric objects
        metrics = copy.deepcopy(self.metrics)

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(data["dataloader"]))
        else:
            total_minibatch = len(data["dataloader"])

        # note that we only use progbar if global rank = 0
        # for other processes, we only use progbar if synchronized_print is false
        if self.use_progress_bar and (
            not self.sync_print or self.FABRIC.is_global_zero
        ):
            loader = tqdm(
                data["dataloader"],
                desc=f"#Evaluating {dataset_name}: ",
                ncols=120,
                ascii=True,
            )
        else:
            loader = data["dataloader"]

        with torch.no_grad():
            for minibatch_idx, (inputs, labels) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                predictions = model(inputs)
                for m in metrics:
                    m.update(predictions=predictions, labels=labels)

        # gather the values from all processes
        # note here that we are collecting the metric object
        # not just the value
        metrics = self.gather_metrics(metrics)

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
            raise RuntimeError(
                "sample_input is None; exporting a model to ONNX requires sample input"
            )

        # get the original module
        if self.FABRIC.world_size == 1:
            module = model._forward_module
        else:
            module = model._forward_module.module

        # then dump it temporarily to disk
        torch.save(module, onnx_path)
        # load again to get a cpu instance
        # we need to avoid exporting the current training instance
        # because it messes up with the training loop and simply
        # makes distributed training stalled
        cpu_model = torch.load(onnx_path, map_location=torch.device("cpu"))
        cpu_model.eval()
        os.remove(onnx_path)

        # now both sample input and cpu model are on cpu, simply export
        if isinstance(sample_input, (list, tuple)):
            # input is a list of tensors
            input_names = ["input_{}".format(idx) for idx in range(len(sample_input))]
        elif isinstance(sample_input, torch.Tensor):
            input_names = ["input"]
        else:
            raise RuntimeError(
                "Invalid model for export. A valid model must accept "
                "a tensor or a list of tensor as inputs"
            )

        with torch.no_grad():
            outputs = cpu_model(sample_input)
            if isinstance(outputs, (list, tuple)):
                for item in outputs:
                    if not isinstance(item, torch.Tensor):
                        raise RuntimeError(
                            "Cannot export model that returns a list of non-tensor"
                        )
                output_names = ["output_{}".format(idx) for idx in range(len(outputs))]
            elif isinstance(outputs, torch.Tensor):
                output_names = ["output"]
            elif isinstance(outputs, dict):
                raise RuntimeError(
                    "Cannot export model that returns a dictionary as outputs"
                )

        dynamic_axes = {}
        if self.onnx_config["dynamic_batch"]:
            for name in input_names + output_names:
                dynamic_axes[name] = {0: "batch_size"}

        torch.onnx.export(
            cpu_model,
            sample_input,
            onnx_path,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        self.logger.info(f"save model in ONNX format in {onnx_path}")


def get_trainer(config: dict, accelerator: str):
    """
    Returns trainer object
    Args:
        config (dict): a dictionary that holds a particular config for this experiment

    Returns:
        a trainer instance
    """

    # create lr scheduler
    if config["lr_scheduler"] == "cosine":
        lr_scheduler = get_cosine_lr_scheduler(config["start_lr"], config["stop_lr"])
    elif config["lr_scheduler"] == "multiplicative":
        lr_scheduler = get_multiplicative_lr_scheduler(
            config["start_lr"],
            config["epochs_to_drop_lr"],  # the epoch number to drop learning rate
            config[
                "lr_multiplicative_factor"
            ],  # factor to multiply to the learning rate to change
        )
    else:
        raise RuntimeError(f"lr scheduler {config['lr_scheduler']} is not supported")

    # TODO: parse loss function and metrics here

    # this basically setups fabric
    Trainer.setup(accelerator=accelerator)

    trainer = Trainer(
        n_epoch=config["n_epoch"],
        output_dir=config["output_dir"],
        loss_function=config["loss_function"],
        metrics=config["metrics"],
        monitor_metric=config["monitor_metric"],
        monitor_direction=config["monitor_direction"],
        checkpoint_idx=config["checkpoint_idx"],
        lr_scheduler=lr_scheduler,
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
        grad_accumulation_step=config["grad_accumulation_step"],
        checkpoint_dir=config["checkpoint_dir"],
        checkpoint_freq=config["checkpoint_freq"],
        max_checkpoint=config["max_checkpoint"],
        eval_freq=config["eval_freq"],
        print_freq=config["print_freq"],
        synchronized_print=config["synchronized_print"],
        use_progress_bar=config["use_progress_bar"],
        test_mode=config["test_mode"],
        retain_metric_objects=config["retain_metric_objects"],
    )
    return trainer
