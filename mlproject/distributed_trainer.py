"""
distributed_trainer.py: base distributed trainer implementation for mlproject
-----------------------------------------------------------------------------


* Copyright: Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-11-18
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""


from __future__ import annotations
from typing import Union
import torch
import torch.optim as optim
import lightning
from tqdm import tqdm
import os
from glob import glob
import pandas as pd
import plotly.express as px
import time
import numpy as np
import tempfile
import dill
from typing import Callable
import copy

from mlproject.metric import Metric


def get_cosine_lr_scheduler(init_lr, final_lr):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = final_lr + 0.5 * (init_lr - final_lr) * (
            1 + np.cos(np.pi * epoch_idx / n_epoch)
        )
        return lr

    return lr_scheduler


def get_multiplicative_lr_scheduler(init_lr, drop_at, multiplicative_factor):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = init_lr
        for epoch in drop_at:
            if epoch_idx + 1 >= epoch:
                lr *= multiplicative_factor
        return lr

    return lr_scheduler


class Logger:
    def __init__(self, id: int):
        self.file_handles = []
        self._id = id

    def add_file_handle(self, fid):
        self.file_handles.append(fid)

    def close(self):
        for fid in self.file_handles:
            fid.close()
        self.file_handles = []

    def timestamp(self):
        return str(pd.Timestamp.now()).split(".")[0]

    def info(self, msg, id=[0]):
        if self._id in id:
            print(f"{self.timestamp()} INFO: Process Global Rank: {self._id} | {msg}\n")
        for fid in self.file_handles:
            fid.write(
                f"{self.timestamp()} INFO: Process Global Rank: {self._id} | {msg}\n"
            )

    def debug(self, msg, id=[0]):
        if self._id in id:
            print(
                f"{self.timestamp()} DEBUG: Process Global Rank: {self._id} | {msg}\n"
            )
        for fid in self.file_handles:
            fid.write(
                f"{self.timestamp()} DEBUG: Process Global Rank: {self._id} | {msg}\n"
            )

    def warning(self, msg, id=[0]):
        if self._id in id:
            print(
                f"{self.timestamp()} WARNING: Process Global Rank: {self._id} | {msg}\n"
            )
        for fid in self.file_handles:
            fid.write(
                f"{self.timestamp()} WARNING: Process Global Rank: {self._id} | {msg}\n"
            )

    def error(self, msg, id=[0]):
        if self._id in id:
            print(
                f"{self.timestamp()} ERROR: Process Global Rank: {self._id} | {msg}\n"
            )
        for fid in self.file_handles:
            fid.write(
                f"{self.timestamp()} ERROR: Process Global Rank: {self._id} | {msg}\n"
            )


class Trainer:
    n_test_minibatch = 100
    FABRIC = None

    @classmethod
    def setup(
        cls,
        accelerator: str = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision=None,
    ):
        cls.FABRIC = lightning.Fabric(
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
        )
        cls.FABRIC.launch()

    def __init__(
        self,
        n_epoch: int,
        output_dir: str,
        loss_function: Callable,
        metrics: list,
        monitor_metric: str,
        monitor_direction: str,
        checkpoint_idx: int = -1,
        lr_scheduler: Callable = get_cosine_lr_scheduler(1e-3, 1e-5),
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
        grad_accumulation_step: int = 1,
        log_dir: str = None,
        checkpoint_freq: int = 10,
        max_checkpoint: int = 10,
        eval_freq: int = 1,
        print_freq: int = 10,
        synchronized_print: bool = True,
        use_progress_bar: bool = True,
        test_mode: bool = False,
        retain_metric_objects: bool = True,
        sample_input=None,
    ):
        if not (isinstance(eval_freq, int) and eval_freq >= 1):
            msg = (
                "eval_freq must be an integer >= 1; ",
                f"received eval_freq={eval_freq}",
            )
            raise RuntimeError("".join(msg))

        if not (isinstance(print_freq, int) and print_freq >= 1):
            msg = (
                "print_freq must be an integer >= 1; ",
                f"received print_freq={print_freq}",
            )
            raise RuntimeError("".join(msg))

        self.output_dir = output_dir
        self.n_epoch = n_epoch
        self.epoch_idx = 0
        self.checkpoint_idx = checkpoint_idx
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.grad_accumulation_step = grad_accumulation_step
        self.log_dir = log_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoint = max_checkpoint
        self.eval_freq = eval_freq
        # print frequency must be at least the grad_accumulation_step
        self.print_freq = max(print_freq, grad_accumulation_step)
        self.sync_print = synchronized_print
        self.use_progress_bar = use_progress_bar
        self.test_mode = test_mode
        self.retain_metric_objects = retain_metric_objects
        self.sample_input = sample_input

        valid = False
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise RuntimeError(
                    f"the given metric {metric} must inherit from mlproject.metrics.Metric class"
                )
            if metric.name() == monitor_metric:
                valid = True

        if not valid:
            raise RuntimeError(
                f"monitor_metric={monitor_metric} does not exist in the list of `metrics`"
            )

        self.history = {}

        self.loss_function = loss_function
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.monitor_direction = monitor_direction

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.final_checkpoint_file = os.path.join(
            self.output_dir, "final_checkpoint.pt"
        )
        self.onnx_checkpoint_file = os.path.join(
            self.output_dir, "final_checkpoint.onnx"
        )
        self.final_performance_file = os.path.join(
            self.output_dir, "final_performance.dill"
        )

    def prepare_log_dir(self):
        if self.log_dir in ["", None]:
            self.log_dir_obj = tempfile.TemporaryDirectory()
            self.log_dir = self.log_dir_obj.name
        else:
            self.log_dir_obj = None

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def has_final_artifacts(self):
        has_pt_ckpt = os.path.exists(self.final_checkpoint_file)
        has_onnx_ckpt = os.path.exists(self.onnx_checkpoint_file)
        has_perf_file = os.path.exists(self.final_performance_file)
        if has_pt_ckpt and has_onnx_ckpt and has_perf_file:
            return True
        else:
            return False

    def verify_data(self, data: dict):
        if data is None:
            return
        msg = (
            "training/validation/test data must be passed as a dictionary that contains ",
            '"dataloader" as key',
        )
        if not isinstance(data, dict) or "dataloader" not in data:
            raise RuntimeError(msg)

    def prepare_data(self, train_data: dict, val_data: dict, test_data: dict):
        """
        Setup for distributed training
        """
        train_data["dataloader"] = self.FABRIC.setup_dataloaders(
            train_data["dataloader"]
        )
        if val_data is not None and val_data["dataloader"] is not None:
            val_data["dataloader"] = self.FABRIC.setup_dataloaders(
                val_data["dataloader"]
            )

        if test_data is not None and test_data["dataloader"] is not None:
            test_data["dataloader"] = self.FABRIC.setup_dataloaders(
                test_data["dataloader"]
            )

    def prepare_model(self, model, optimizer):
        # fabric prep
        model, optimizer = self.FABRIC.setup(model, optimizer)
        # then make sure params of optimizer are on correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.FABRIC.device)

        return model, optimizer

    def fit(
        self,
        model: torch.nn.Module,
        train_data: dict,
        val_data: dict = None,
        test_data: dict = None,
        tensorboard_logger=None,
        logger_prefix="",
    ):
        if self.FABRIC is None:
            raise RuntimeError(
                (
                    f"{self.__class__}.setup() must be called before fit(). "
                    "Ideally before any dataset or dataloader creation"
                )
            )
        self.verify_data(train_data)
        self.verify_data(val_data)
        self.verify_data(test_data)

        self.prepare_data(train_data, val_data, test_data)

        if self.test_mode:
            self.total_minibatch = min(
                self.n_test_minibatch, len(train_data["dataloader"])
            )
        else:
            self.total_minibatch = len(train_data["dataloader"])

        self.total_train_minibatch = self.n_epoch * self.total_minibatch

        # look for final checkpoint
        if self.has_final_artifacts():
            self.logger.warning(f"final artifacts exist under {self.output_dir}")
            self.logger.warning("no training was done")
            self.logger.warning("if you want to retrain model, please remove them")

            # load performance
            with open(self.final_performance_file, "rb") as fid:
                performance = dill.load(fid)

            return performance

        self.prepare_log_dir()

        model.float()
        optimizer = self.get_optimizer(model)

        # note that load_from_checkpoint assume model is in CPU
        # model should be moved to GPU only after this
        self.load_from_checkpoint(model, optimizer)

        # prepare model and optimizer for distributed training
        # here model is moved to correct device, optimizer params are also moved correctly
        model, optimizer = self.prepare_model(model, optimizer)

        self.start_time = time.time()
        self.start_minibatch_idx = self.cur_minibatch

        while self.epoch_idx < self.n_epoch:
            # optimize one epoch
            self.update_lr(optimizer)
            self.update_loop(model, train_data, optimizer)
            self.epoch_idx += 1

            if (self.epoch_idx % self.eval_freq) == 0:
                train_performance = self.eval(model, train_data, "train set")
                val_performance = self.eval(model, val_data, "val set")
                test_performance = self.eval(model, test_data, "test set")
                self.update_metrics(
                    train_performance, val_performance, test_performance
                )

                self.print_metrics(train_performance, "train")
                self.print_metrics(val_performance, "val")
                self.print_metrics(test_performance, "test")

                # if checkpoint frequency is None, then save checkpoint after
                # evaluation
                if self.checkpoint_freq is None:
                    self.update_checkpoint(model, optimizer, False)
                    # remember to to convert model to train stage
                    model.train()

        # load the best model based on validation performance if exist, or train performance
        self.load_best(model)

        # eval this best model
        self.logger.info("evaluating performance of the final model...")
        final_train_metrics = self.eval(model, train_data, "train set")
        final_val_metrics = self.eval(model, val_data, "val set")
        final_test_metrics = self.eval(model, test_data, "test set")

        self.print_metrics(final_train_metrics, "final_train")
        self.print_metrics(final_val_metrics, "final_val")
        self.print_metrics(final_test_metrics, "final_test")

        # clean up temp dir
        if self.log_dir_obj is not None:
            self.log_dir_obj.cleanup()

        performance_curves = {}
        for key, value_list in self.history.items():
            if isinstance(value_list[0], (int, float)):
                performance_curves[key] = value_list
            else:
                performance_curves[key] = [
                    metric_object.value() for metric_object in value_list
                ]

        # close the file handle of loss curve
        self.loss_curve.close()
        self.visualize_performance(performance_curves)

        performance = {
            "history": self.history,
            "train": final_train_metrics,
            "val": final_val_metrics,
            "test": final_test_metrics,
        }

        # save performance
        with open(self.final_performance_file, "wb") as fid:
            dill.dump(performance, fid, recurse=True)

        # save torch checkpoint
        if self.FABRIC.is_global_zero:
            torch.save(model, self.final_checkpoint_file)
            self.logger.info(f"save final model in {self.final_checkpoint_file}")
        self.FABRIC.barrier()

        # save onnx checkpoint
        self.export_to_onnx(model, self.sample_input, self.onnx_checkpoint_file)

        if isinstance(self.logger, Logger):
            self.logger.close()

        return performance

    def export_to_onnx(self, model, sample_input, onnx_path):
        if sample_input is None:
            raise RuntimeError(
                "sample_input is None; exporting a model to ONNX requires sample input"
            )

        # get the original module
        if self.FABRIC.world_size == 1:
            module = model._fabric_module
        else:
            module = self.FABRIC._fabric_module.module

        # then dump it temporarily to disk
        torch.save(module, onnx_path)
        # load again to get a cpu instance
        # we need to avoid exporting the current training instance
        # because it messes up with the training loop and simply
        # makes distributed training stalled
        cpu_model = torch.load(onnx_path, map_location=torch.device("cpu"))
        os.remove(onnx_path)

        # now both sample input and cpu model are on cpu, simply export
        torch.onnx.export(
            cpu_model,
            sample_input,
            onnx_path,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        self.logger.info(f"save model in ONNX format in {onnx_path}")

    def visualize_performance(self, performance):
        """
        visualize training curves
        """
        # now find the names of all metrics in performance
        metric_names = set(
            [
                x.replace("train_", "").replace("val_", "").replace("test_", "")
                for x in performance.keys()
            ]
        )

        for metric in metric_names:
            plot_metrics = {"name": [], "epoch": [], "metric_value": []}
            x_ticks = self.history_indices
            nb_x_points = len(x_ticks)

            # train curve
            for idx in range(nb_x_points):
                plot_metrics["name"].append("train_" + metric)
                plot_metrics["epoch"].append(x_ticks[idx])
                plot_metrics["metric_value"].append(performance["train_" + metric][idx])

            # val curve
            if "val_" + metric in performance:
                for idx in range(nb_x_points):
                    plot_metrics["name"].append("val_" + metric)
                    plot_metrics["epoch"].append(x_ticks[idx])
                    plot_metrics["metric_value"].append(
                        performance["val_" + metric][idx]
                    )

            # test curve
            if "test_" + metric in performance:
                for idx in range(nb_x_points):
                    plot_metrics["name"].append("test_" + metric)
                    plot_metrics["epoch"].append(x_ticks[idx])
                    plot_metrics["metric_value"].append(
                        performance["test_" + metric][idx]
                    )

            df = pd.DataFrame(plot_metrics)
            fig = px.line(df, x="epoch", y="metric_value", color="name", title=metric)
            filename = os.path.join(self.output_dir, f"{metric}.html")
            fig.write_html(filename)

        # plot performance curve
        loss_curve_file = os.path.join(self.output_dir, "loss_curve.txt")
        with open(loss_curve_file, "r") as fid:
            data = fid.read().split("\n")[:-1]
            loss_curve = []
            minibatch_indices = []
            for item in data:
                loss_curve.append(float(item.split(",")[1]))
                minibatch_indices.append(int(item.split(",")[0]))

        df = pd.DataFrame(
            {
                "minibatch": minibatch_indices,
                "loss value": loss_curve,
                "name": "train loss",
            }
        )
        fig = px.line(
            df, x="minibatch", y="loss value", color="name", title="train loss curve"
        )
        filename = os.path.join(self.output_dir, "loss_curve.html")
        fig.write_html(filename)

    def get_metric_value(self, metric):
        if isinstance(metric, (int, float)):
            return metric
        elif hasattr(metric, "value") and callable(metric.value):
            return metric.value()
        else:
            raise RuntimeError("Cannot access metric value")

    def load_best(self, model):
        # load the best model from checkpoints based on monitor_metric
        has_val = "val_" + self.monitor_metric in self.history

        val_key = "val_" + self.monitor_metric
        train_key = "train_" + self.monitor_metric

        if has_val and val_key in self.history and len(self.history[val_key]) > 0:
            best_value = self.history[val_key][-1]
            self.logger.info(
                "loading the best checkpoint based on performance measured on validation data"
            )
        else:
            self.logger.info(
                "loading the best checkpoint based on performance measured on train data"
            )
            if train_key in self.history and len(self.history[train_key]) > 0:
                best_value = self.history[train_key][-1]
            else:
                return

        best_value = self.get_metric_value(best_value)
        state_dict = model.state_dict()

        checkpoint_files = [
            os.path.join(self.log_dir, f)
            for f in os.listdir(self.log_dir)
            if f.startswith("checkpoint_")
        ]
        history_files = [
            os.path.join(self.log_dir, f)
            for f in os.listdir(self.log_dir)
            if f.startswith("history_")
        ]

        for checkpoint_filename, history_filename in zip(
            checkpoint_files, history_files
        ):
            with open(history_filename, "rb") as fid:
                history = dill.load(fid)

            if (
                has_val
                and val_key in history["history"]
                and len(history["history"][val_key]) > 0
            ):
                metric_value = history["history"][val_key][-1]
            elif (
                train_key in history["history"]
                and len(history["history"][train_key]) > 0
            ):
                metric_value = history["history"][train_key][-1]
            else:
                continue

            metric_value = self.get_metric_value(metric_value)

            if (self.monitor_direction == "lower" and metric_value < best_value) or (
                self.monitor_direction == "higher" and metric_value > best_value
            ):
                best_value = metric_value
                checkpoint = torch.load(
                    checkpoint_filename, map_location=self.FABRIC.device
                )
                state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(state_dict)

    def get_optimizer(self, model):
        assert self.optimizer in [
            "adam",
            "sgd",
            "adamW",
            "rmsprop",
        ], 'Given optimizer "{}" is not supported'.format(self.optimizer)

        # get current learning rate
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)

        # get separate batchnorm parameters and other parameters
        # if .get_parameters() is implemented in the model
        if hasattr(model, "get_parameters") and callable(model.get_parameters):
            bn_params, other_params = model.get_parameters()

            if len(bn_params) > 0:
                params = [
                    {"params": bn_params, "weight_decay": 0},
                    {"params": other_params, "weight_decay": self.weight_decay},
                ]
            else:
                params = [{"params": other_params, "weight_decay": self.weight_decay}]

            if self.optimizer == "adam":
                optimizer = optim.Adam(params, lr=lr)
            elif self.optimizer == "sgd":
                optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
            elif self.optimizer == "adamW":
                optimizer = optim.AdamW(params, lr=lr)
            elif self.optimizer == "rmsprop":
                optimizer = optim.RMSprop(params, lr=lr, weight_decay=self.weight_decay)
            else:
                raise NotImplemented
        else:
            if self.optimizer == "adam":
                optimizer = optim.Adam(
                    model.parameters(), weight_decay=self.weight_decay, lr=lr
                )
            elif self.optimizer == "sgd":
                optimizer = optim.SGD(
                    model.parameters(),
                    weight_decay=self.weight_decay,
                    lr=lr,
                    momentum=0.9,
                    nesterov=True,
                )
            elif self.optimizer == "adamW":
                optimizer = optim.AdamW(
                    model.parameters(), weight_decay=self.weight_decay, lr=lr
                )
            elif self.optimizer == "rmsprop":
                optimizer = optim.RMSprop(
                    model.parameters(), weight_decay=self.weight_decay, lr=lr
                )
            else:
                raise NotImplementedError(f"Optimizer {self.optimizer} not supported")

        return optimizer

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

        if self.use_progress_bar:
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

    def gather_metrics(self, metrics: list):
        """
        Gather metric objects from all processes
        """

        if self.FABRIC.world_size == 1:
            return metrics

        for m in metrics:
            # dump the metric content to dictionary
            m_content = m.dump()
            # gather metric content from all processes
            values = self.FABRIC.all_gather(m_content)

            # if any field is not int, float or tensor, it will not be gathered
            # so we need to replace it
            for k, v in values.items():
                if not isinstance(v, torch.Tensor):
                    values[k] = [
                        m_content[k],
                    ] * self.FABRIC.world_size
            # values is a dictionary, each key holds a list of value
            # now unwrap into a list of dictionaries
            fields = list(values.keys())
            # then reconstruct metric objects from all processes
            instances = []
            for i in range(self.FABRIC.world_size):
                metric_obj = copy.deepcopy(m)
                serialized_values = {}
                for field in fields:
                    serialized_values[field] = values[field][i]
                metric_obj.load(serialized_values)
                instances.append(metric_obj)

            # reset the metric object
            m.reset()
            # then gather them
            m.merge(*instances)

        return metrics

    def update_lr(self, optimizer):
        # update learning rate using lr_scheduler
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
            if (self.cur_minibatch + 1 % self.grad_accumulation_step) == 0:
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

    def update_checkpoint(self, model, optimizer, epoch_ended):
        # only save checkpoint if global rank is zero
        if self.FABRIC.is_global_zero:
            print(f"process id running checkpoint saving: {self.FABRIC.global_rank}")
            # put in eval mode
            model.eval()

            # checkpoint every K minibatch
            checkpoint = {
                "epoch_idx": self.epoch_idx,
                "cur_minibatch": self.cur_minibatch,
                "minibatch_idx": self.minibatch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_ended": epoch_ended,
            }
            history = {"history": self.history, "history_indices": self.history_indices}

            checkpoint_file = os.path.join(
                self.log_dir,
                "checkpoint_{:09d}_{:09d}.pt".format(
                    self.cur_minibatch, self.epoch_idx
                ),
            )

            torch.save(checkpoint, checkpoint_file)

            # need to separate history because it contains metric objects, which may require
            # dill for serialization
            history_file = os.path.join(
                self.log_dir,
                "history_{:09d}_{:09d}.dill".format(self.cur_minibatch, self.epoch_idx),
            )
            with open(history_file, "wb") as fid:
                dill.dump(history, fid, recurse=True)

            self.logger.info(f"save checkpoint to {checkpoint_file}")
            self.logger.info(f"save history to {history_file}")

            checkpoint_files = glob(self.log_dir + "/checkpoint_*.pt")
            checkpoint_files.sort()
            history_files = glob(self.log_dir + "/history_*.dill")
            history_files.sort()

            no_checkpoint_files = len(checkpoint_files)

            if self.max_checkpoint > 0 and no_checkpoint_files > self.max_checkpoint:
                for idx in range(0, no_checkpoint_files - self.max_checkpoint):
                    os.remove(checkpoint_files[idx])
                    os.remove(history_files[idx])

            # ----- handle onnx checkpoints --------------------------
            onnx_file = os.path.join(
                self.log_dir,
                "model_{:09d}_{:09d}.onnx".format(self.cur_minibatch, self.epoch_idx),
            )
            self.export_to_onnx(model, self.sample_input, onnx_file)
            onnx_files = glob(self.log_dir + "/model_*.onnx")
            onnx_files.sort()
            nb_onnx_file = len(onnx_files)
            if self.max_checkpoint > 0 and nb_onnx_file > self.max_checkpoint:
                for idx in range(0, nb_onnx_file - self.max_checkpoint):
                    os.remove(onnx_files[idx])

        # barrier is called for every process
        print(f"process id before barrier: {self.FABRIC.global_rank}")
        self.FABRIC.barrier()
        print(f"process id after barrier: {self.FABRIC.global_rank}")

    def print_and_update(
        self,
        total_time,
        data_prep_time,
        forward_time,
        backward_time,
        profile_counter,
        optimizer_time,
        optimizer_step,
    ):
        if self.sync_print:
            # if we want to sync loss result accross processes then print
            # this option can slow down training
            avg_loss = self.accumulated_loss / self.accumulated_loss_counter
            avg_loss = self.FABRIC.all_reduce(avg_loss)
            if self.FABRIC.is_global_zero:
                self.loss_curve.write("{},{}\n".format(self.cur_minibatch, avg_loss))
        else:
            # otherwise, all processes write to the same loss curve file
            # we will accumulate and reduce when plotting the loss curve
            avg_loss = self.accumulated_loss / self.accumulated_loss_counter
            self.loss_curve.write("{},{}\n".format(self.cur_minibatch, avg_loss))

        # find number of digits of total train minibatch
        nb_digit = len(str(self.total_train_minibatch))
        formatter = "{:0" + str(nb_digit) + "d}"
        cur_minibatch = formatter.format(self.cur_minibatch)

        nb_digit = len(str(self.epoch_idx + 1))
        formatter = "{:0" + str(nb_digit) + "d}"
        epoch = formatter.format(self.epoch_idx + 1)

        print_content = [
            "minibatch: {} / {}| ".format(cur_minibatch, self.total_train_minibatch),
            "epoch: {}| ".format(epoch),
            "loss: {}| ".format(avg_loss),
            "avg latency per minibatch (in seconds): all={:.3f}, ".format(
                total_time / profile_counter
            ),
            "data prep={:.3f}, ".format(data_prep_time / profile_counter),
        ]

        print_content.append(
            "forward={:.3f}, ".format(forward_time / profile_counter),
        )
        print_content.append(
            "backward={:.3f}, ".format(backward_time / profile_counter),
        )
        if optimizer_step > 0:
            print_content.append(
                "optimizer step={:.3f}, ".format(optimizer_time / optimizer_step),
            )

        estimated_time = self.get_estimated_time()
        print_content.append(
            "time taken: {:02d}:{:02d}:{:02d}| ".format(
                estimated_time["hour_taken"],
                estimated_time["minute_taken"],
                estimated_time["second_taken"],
            )
        )
        print_content.append(
            "time left: {:02d}:{:02d}:{:02d}| ".format(
                estimated_time["hour_left"],
                estimated_time["minute_left"],
                estimated_time["second_left"],
            )
        )

        if self.sync_print:
            self.logger.info("".join(print_content), id=[0])
        else:
            self.logger.info("".join(print_content), id=[self.FABRIC.global_rank])

        self.accumulated_loss_counter = 0
        self.accumulated_loss = 0

    def update_metrics(
        self, train_metrics: dict, val_metrics: dict, test_metrics: dict
    ):
        # update indices
        self.history_indices.append(self.epoch_idx)

        # add new values to the history list
        prefixes = ["train_", "val_", "test_"]
        values = [train_metrics, val_metrics, test_metrics]
        for prefix, value in zip(prefixes, values):
            for metric_name in value.keys():
                if prefix + metric_name in self.history:
                    self.history[prefix + metric_name].append(value[metric_name])
                else:
                    self.history[prefix + metric_name] = [value[metric_name]]

    def print_metrics(self, metrics, prefix):
        names = list(metrics.keys())
        names.sort()
        for name in names:
            value = self.get_metric_value(metrics[name])
            self.logger.info("{} {}: {:.6f}".format(prefix, name, value))

    def get_estimated_time(self):
        minibatch_left = self.total_train_minibatch - self.cur_minibatch
        minibatch_taken = self.cur_minibatch - self.start_minibatch_idx
        time_taken = time.time() - self.start_time
        time_left = int(time_taken * minibatch_left / minibatch_taken)

        hour_left = int(time_left / 3600)
        minute_left = int((time_left - hour_left * 3600) / 60)
        second_left = int((time_left - hour_left * 3600 - minute_left * 60))

        hour_taken = int(time_taken / 3600)
        minute_taken = int((time_taken - hour_taken * 3600) / 60)
        second_taken = int((time_taken - hour_taken * 3600 - minute_taken * 60))
        return {
            "hour_taken": hour_taken,
            "minute_taken": minute_taken,
            "second_taken": second_taken,
            "hour_left": hour_left,
            "minute_left": minute_left,
            "second_left": second_left,
        }

    def update_tensorboard(self, tensorboard_logger, logger_prefix):
        names = list(self.history.keys())
        names.sort()
        if tensorboard_logger is not None:
            for name in names:
                if len(self.history[name]) > 0:
                    value = self.history[name][-1]
                    value = self.get_metric_value(value)
                    tensorboard_logger.add_scalar(
                        tag="{}/{}".format(logger_prefix, name),
                        scalar_value=value,
                        global_step=self.epoch_idx + 1,
                    )
                    tensorboard_logger.flush()

    def load_from_checkpoint(self, model, optimizer):
        ckp_files = [
            os.path.join(self.log_dir, f)
            for f in os.listdir(self.log_dir)
            if f.startswith("checkpoint_")
        ]
        ckp_files.sort()
        history_files = [
            os.path.join(self.log_dir, f)
            for f in os.listdir(self.log_dir)
            if f.startswith("history_")
        ]
        history_files.sort()

        if self.checkpoint_idx == -1:
            # load from latest checkpoint
            if len(ckp_files) > 0:
                checkpoint = torch.load(ckp_files[-1], map_location=torch.device("cpu"))
                with open(history_files[-1], "rb") as fid:
                    history = dill.load(fid)
            else:
                checkpoint = None
                history = None
        elif self.checkpoint_idx is None:
            # train from scratch
            checkpoint = None
            history = None
        elif self.checkpoint_idx >= 0:
            if len(ckp_files) == 0:
                self.logger.warning(
                    "no checkpoint exists; training will be done from scratch"
                )
                checkpoint = None
                history = None
            else:
                if self.checkpoint_idx < len(ckp_files):
                    ckp_file = ckp_files[self.checkpoint_idx]
                    checkpoint = torch.load(ckp_file, map_location=torch.device("cpu"))
                    self.logger.info(f"checkpoint file {ckp_file} loaded")
                    with open(history_files[self.checkpoint_idx], "rb") as fid:
                        history = dill.load(fid)
                    self.logger.info(f"history file {ckp_file} loaded")
                else:
                    msg = (
                        f"invalid checkpoint index: {self.checkpoint_idx}; ",
                        f"there are only {len(ckp_files)} checkpoints",
                    )
                    raise RuntimeError("".join(msg))
        else:
            raise RuntimeError(f"unknown checkpoint index: {self.checkpoint_idx}")

        self.accumulated_loss_counter = 0
        self.accumulated_loss = 0
        self.loss_curve = open(os.path.join(self.output_dir, "loss_curve.txt"), "a")

        if checkpoint is None:
            self.epoch_idx = 0
            self.history = {}
            self.history_indices = []
            self.cur_minibatch = 0
            self.minibatch_idx = 0
        else:
            # set the epoch index and previous metric values
            self.cur_minibatch = checkpoint["cur_minibatch"]
            self.minibatch_idx = checkpoint["minibatch_idx"]
            self.epoch_idx = checkpoint["epoch_idx"]
            self.history = history["history"]
            self.history_indices = history["history_indices"]

            if checkpoint["epoch_ended"]:
                self.epoch_idx += 1
                self.minibatch_idx = 0

            # load model state dict
            model.load_state_dict(checkpoint["model_state_dict"])

            # load optimizer state dict
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.logger = Logger(id=self.FABRIC.global_rank)
        progress_file_handle = open(os.path.join(self.output_dir, "progress.txt"), "a")
        self.logger.add_file_handle(progress_file_handle)
