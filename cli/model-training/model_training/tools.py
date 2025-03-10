"""
tools.py: tools for model training
----------------------------------


* Copyright: 2024 datsbot.com
* Authors: Dat Tran (hello@dats.bio)
* Date: 2024-08-04
* Version: 0.0.1


This is part of model-training package

License
-------
Proprietary License

"""

from __future__ import annotations
import os
import shutil
import dill
import time
from torch.utils.data import Dataset as TorchDataset
from mlproject.config import create_all_config as _create_all_config
from mlproject.config import get_config_string
from mlproject.distributed_trainer import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler,
    Trainer as BaseTrainer,
    Logger,
)
from model_training.template.config import ALL_CONFIGS as TEMPLATE_CONFIG
from swift_loader import SwiftLoader
import importlib.util
import pprint
import random
import string
import sys
import traceback
import torch


def load_module(module_file, attribute, module_name=None):
    if module_name is None:
        module_name = "".join(random.sample(string.ascii_letters, 10))

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException as error1:
        try:
            # try to append directory that contains module_file to path
            print(f"fails to import attribute {attribute} from module {module_file}")
            module_path = os.path.dirname(os.path.abspath(module_file))
            print(f"trying to append {module_path} to sys.path to fix this issue")
            sys.path.append(module_path)

            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except BaseException:
            traceback.print_exc()
            raise error1

    if hasattr(module, attribute):
        return getattr(module, attribute)
    else:
        raise ImportError(
            f"Cannot find attribute {attribute} in the given module at {module_file}"
        )


class Dataset(TorchDataset):
    def __init__(self, path: str, name: str, arguments: dict):
        constructor = load_module(path, name)
        self._data = constructor(**arguments)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class Trainer(BaseTrainer):
    def fit(
        self,
        model: torch.nn.Module,
        train_data: dict,
        val_data: dict = None,
        test_data: dict = None,
        tensorboard_logger=None,
        logger_prefix="",
        load_best: bool = True,
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
            if self.FABRIC.is_global_zero:
                print(
                    f"Final artifacts exist under {self.output_dir}. "
                    "No training was done. "
                    "If you want to retrain model, please remove artifacts"
                )

            # load performance
            with open(self.final_performance_file, "rb") as fid:
                performance = dill.load(fid)

            return performance

        self.prepare_checkpoint_dir()

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

        # move loss function
        if isinstance(self.loss_function, torch.nn.Module):
            self.loss_function.to(self.FABRIC.device)

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
        if load_best:
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
        if self.checkpoint_dir_obj is not None:
            self.checkpoint_dir_obj.cleanup()

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

        # get the original module
        if self.FABRIC.world_size == 1:
            module = model._forward_module
        else:
            module = model._forward_module.module

        # save torch checkpoint
        if self.FABRIC.is_global_zero:
            torch.save(module.state_dict(), self.final_checkpoint_file)
            self.logger.info(
                f"save final model state dict in {self.final_checkpoint_file}"
            )
        self.FABRIC.barrier()

        # save onnx checkpoint, only rank 0 does this
        if self.FABRIC.is_global_zero:
            if self.sample_input is None:
                for inputs, labels in train_data["dataloader"]:
                    self.sample_input = inputs.cpu()
                    break
            self.export_to_onnx(model, self.sample_input, self.onnx_checkpoint_file)
        self.FABRIC.barrier()

        if isinstance(self.logger, Logger):
            self.logger.close()

        return performance

    # need to overwrite onnx export because of dynamic model import
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

        weight_path = onnx_path.replace(".onnx", ".pt")

        # then dump it temporarily to disk
        torch.save(module.state_dict(), weight_path)
        # load again to get a cpu instance
        # we need to avoid exporting the current training instance
        # because it messes up with the training loop and simply
        # makes distributed training stalled
        weights = torch.load(weight_path, map_location=torch.device("cpu"))
        cpu_model = self.model_constructor(**self.model_arguments)
        cpu_model.load_state_dict(weights)
        cpu_model.eval()
        os.remove(weight_path)

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

        # export with dynamic batch size
        dynamic_axes = {}
        for name in input_names + output_names:
            dynamic_axes[name] = {0: "batch_size"}

        dynamic_onnx_path = onnx_path.replace(".onnx", "_bs=dynamic.onnx")
        fixed_onnx_path = onnx_path.replace(".onnx", "_bs=1.onnx")

        torch.onnx.export(
            cpu_model,
            sample_input,
            dynamic_onnx_path,
            opset_version=self.opset,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        self.logger.info(
            f"save model with dynamic batch size in ONNX format in {dynamic_onnx_path}"
        )

        # export with batch size = 1
        sample_input = sample_input[0:1]
        torch.onnx.export(
            cpu_model,
            sample_input,
            fixed_onnx_path,
            opset_version=self.opset,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={},
        )
        self.logger.info(
            f"save model with batch size = 1 in ONNX format in {fixed_onnx_path}"
        )
        del cpu_model
        del weights


def get_dataset(config: dict, set_name: str):
    constructor_name = f"get_{set_name}_set"
    # try to load, if missing then simply return None
    try:
        load_module(config["dataset"]["implementation"], constructor_name)
    except ImportError:
        # if train set, then throw
        if set_name == "train":
            msg = f"Cannot find get_train_set() method in {config['dataset']['implementation']}"
            raise ImportError(msg)
        else:
            print(
                f"WARNING: Cannot find get_{set_name}_set() method in {config['dataset']['implementation']}. "
                "Ignoring this data"
            )
            return None

    dataset_kwargs = {
        "path": config["dataset"]["implementation"],
        "name": constructor_name,
        "arguments": config["dataset"][f"{set_name}_arguments"],
    }

    return Dataset(**dataset_kwargs)


def get_data_loader(config: dict, set_name: str):
    constructor_name = f"get_{set_name}_set"
    # try to load, if missing then simply return None
    try:
        load_module(config["dataset"]["implementation"], constructor_name)
    except ImportError:
        # if train set, then throw
        if set_name == "train":
            msg = f"Cannot find get_train_set() method in {config['dataset']['implementation']}"
            raise ImportError(msg)
        else:
            print(
                f"WARNING: Cannot find get_{set_name}_set() method in {config['dataset']['implementation']}. "
                "Ignoring this data"
            )
            return None

    dataset_kwargs = {
        "path": config["dataset"]["implementation"],
        "name": constructor_name,
        "arguments": config["dataset"][f"{set_name}_arguments"],
    }

    loader = SwiftLoader(
        dataset_class=Dataset,
        dataset_kwargs=dataset_kwargs,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        nb_consumer=config["dataloader"]["nb_consumer"],
        worker_per_consumer=config["dataloader"]["worker_per_consumer"],
        logger=os.path.join(config["log_dir"], set_name),
    )
    return loader


def dispose_data_loader(*args):
    """
    closing swift loader
    """
    for item in args:
        if item is not None and hasattr(item, "close") and callable(item.close):
            item.close()


def get_trainer(config: dict, accelerator: str):
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

    # this basically setups fabric
    Trainer.setup(accelerator=accelerator)

    get_loss_function = load_module(
        config["loss"]["implementation"], "get_loss_function"
    )
    loss_function = get_loss_function(**config["loss"]["arguments"])

    # metric
    metric_src_path = config["metric"]["implementation"]
    get_metric = load_module(metric_src_path, "get_metric")
    metric_info = get_metric(**config["metric"]["arguments"])
    for key in ["metrics", "monitor_metric", "monitor_direction"]:
        if key not in metric_info:
            raise RuntimeError(
                f"get_metric(**kwargs: dict) in {metric_src_path} should return a dict with 3 keys:"
                "metrics, monitor_metric, monitor_direction"
            )

    trainer = Trainer(
        n_epoch=config["n_epoch"],
        output_dir=config["output_dir"],
        loss_function=loss_function,
        metrics=metric_info["metrics"],
        monitor_metric=metric_info["monitor_metric"],
        monitor_direction=metric_info["monitor_direction"],
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

    trainer.model_constructor = load_module(
        config["model"]["implementation"], "get_model"
    )
    trainer.model_arguments = config["model"]["arguments"]
    trainer.opset = config.get("onnx_opset", 11)

    return trainer


def get_model(config: dict):
    model_constructor = load_module(config["model"]["implementation"], "get_model")
    model = model_constructor(**config["model"]["arguments"])
    return model


def create_all_config(config_module):
    all_configs = _create_all_config(config_module.ALL_CONFIGS)
    return all_configs


def get_config_size(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    validate_config_module(config_module, path)
    return len(create_all_config(config_module))


def print_all_config(config_module):
    configs = create_all_config(config_module)
    for idx, item in enumerate(configs):
        msg = f"config_index={idx}"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
        pprint.pprint(item)


def get_config(config_module, index):
    # compute all configurations first
    configs = create_all_config(config_module)
    if index < len(configs):
        return configs[index]
    else:
        print(f"number of configurations: {len(configs)}")
        print(f"configuration index {index}")
        raise RuntimeError("the given index is out of range from all configurations")


def validate_config_module(config_module, path):
    required_fields = ["NAME", "DESC", "ALL_CONFIGS"]
    for field in required_fields:
        if not hasattr(config_module, field):
            raise RuntimeError(f"config file {path} is missing field: {field}")

    """
    Check if ALL_CONFIGS have required keys
    """
    for field in TEMPLATE_CONFIG.keys():
        if field not in config_module.ALL_CONFIGS:
            raise RuntimeError(
                f"config file {path} is missing field: {field} in ALL_CONFIGS"
            )


def load_config(file: str, index: str, test_mode: bool) -> tuple[dict, str, str]:
    """
    load configuration values from a given config file
    a config file contains all different experiment combinations
    index refers to the index of the experiment setting to run
    """
    spec = importlib.util.spec_from_file_location("config_module", file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    validate_config_module(config_module, file)

    config = get_config(config_module, index)
    config_name = config_module.NAME
    config_description = config_module.DESC
    config["test_mode"] = test_mode
    config["config_name"] = config_name
    config["config_description"] = config_description
    config["config_file"] = file
    config["config_index"] = index
    return config


def print_config(config: dict):
    print(f"CONFIGURATION NAME: {config['config_name']}")
    print(f"CONFIGURATION DESCRIPTION: {config['config_description']}")
    config_string = get_config_string(config)
    print(config_string)


def prepare_directories(config: dict):
    # prep output dir and log dir
    # note this is the top-level dir for all experiments
    os.makedirs(config["output_dir"], exist_ok=True)

    # need to create subdir as output dir and log dir for this particular exp
    config_name = config["config_name"].replace(" ", "_")
    config_index = config["config_index"]
    config_file = config["config_file"]
    # output directory path has the following structure:
    # user_given_dir/config_name/config_index/
    output_dir = os.path.join(
        config["output_dir"],
        config_name,
        "config_index={:09d}".format(config_index),
    )
    # under the output dir is another subdir for checkpoints
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # now copy the original config file, the config index and the content of
    # the config to log and output dir for reference purposes
    shutil.copy(
        config_file, os.path.join(checkpoint_dir, os.path.basename(config_file))
    )
    shutil.copy(config_file, os.path.join(output_dir, os.path.basename(config_file)))
    with open(
        os.path.join(checkpoint_dir, f"config_index_{config_index}.dill"), "wb"
    ) as fid:
        dill.dump(config, fid)
    with open(
        os.path.join(output_dir, f"config_index_{config_index}.dill"), "wb"
    ) as fid:
        dill.dump(config, fid)

    with open(os.path.join(output_dir, "config_content.txt"), "w") as fid:
        fid.write(get_config_string(config))

    # put directories into config
    config["output_dir"] = output_dir
    config["checkpoint_dir"] = checkpoint_dir
    config["log_dir"] = log_dir
