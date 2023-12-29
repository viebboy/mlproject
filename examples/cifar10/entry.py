"""
entry.py: main training orchestration
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
import argparse
import importlib.util
import os
import shutil
import dill
import copy
import torch
import traceback
from mlproject.config import get_config_string, get_repo_info

from data import get_torch_loader, get_swift_loader, dispose_data_loader
from trainer import get_trainer
from models import DenseNet

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# this should list the supported keyword arguments on cli to overwrite the config
# key of the dictionary should be one of the key in the config dictionary passed to main()
# value should be the function that maps from string to the target type of key
# for example, if we want to overwrite nb_epoch during runtime via cli, then we can add
# SUPPORTED_KWARGS = {"nb_epoch": int}
# when calling entry.py, we can pass number of epoch on-the-fly like this:
# python3 entry.py --index some_number ... nb_epoch=some_number
# parse_args() will parse the arguments and convert the string to the target type, which is int
SUPPORTED_KWARGS = {"n_epoch": int}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Entry script to launch experiment")
    parser.add_argument("--index", required=True, type=int, help="config index to run")

    # these are used to versioned default configs
    parser.add_argument(
        "--config-path",
        default=os.path.join(CUR_DIR, "configs", "config.py"),
        type=str,
        help=(
            "Path to config file, should be a valid .py file. "
            "Default to config.py in the same source code"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        type=str,
        help="Device to use for computation",
    )
    parser.add_argument(
        "--test-mode",
        default="False",
        type=str,
        choices=["false", "False", "True", "true"],
        help="Whether to run in test mode with a small fraction of data",
    )

    parser.add_argument(
        "kwargs",
        nargs="*",
        help=(
            "Additional keyword arguments in the form key=value. "
            f"Supported kwargs include: {SUPPORTED_KWARGS}"
        ),
    )

    # process args
    args = parser.parse_args()
    if args.test_mode in ["True", "true"]:
        args.test_mode = True
    else:
        args.test_mode = False

    cli_kwargs = {}
    for kwarg in args.kwargs:
        key, value = kwarg.split("=")
        if key not in SUPPORTED_KWARGS:
            raise ValueError(f"Unsupported kwarg {key}")

        try:
            value = SUPPORTED_KWARGS[key](value)
        except BaseException:
            traceback.print_exc()
            raise ValueError(
                f"Failed to convert value {value} using type formatter: {SUPPORTED_KWARGS[key]}"
            )
        cli_kwargs[key] = value

    args.kwargs = cli_kwargs
    return args


def load_config(file: str, index: str, test_mode: bool) -> tuple[dict, str, str]:
    """
    load configuration values from a given config file
    a config file contains all different experiment combinations
    index refers to the index of the experiment setting to run
    """
    spec = importlib.util.spec_from_file_location("config_module", file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.get_config(index)
    config_name = config_module.get_name()
    config_description = config_module.get_description()
    config["test_mode"] = test_mode
    config["config_name"] = config_name
    config["config_description"] = config_description
    config["config_file"] = file
    config["config_index"] = index

    repo_info = get_repo_info(os.path.abspath(__file__))
    config["git_url"] = repo_info["git_url"]
    config["git_branch"] = repo_info["git_branch"]
    config["git_commit"] = repo_info["git_commit"]
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
    # user_given_dir/config_name/config_index/trial_index/
    output_dir = os.path.join(
        config["output_dir"],
        config_name,
        "config_index={:09d}".format(config_index),
        "trial_index={:09d}".format(config["trial_index"]),
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


def main(
    exp_config: dict,
    device: str,
    nb_consumer: int,
    cli_kwargs: dict,
) -> None:
    # overwrite options from cli
    for key, value in cli_kwargs.items():
        exp_config[key] = value

    # print config
    if "MLPROJECT_MAIN_PROCESS" not in os.environ:
        # only print on main process
        print_config(exp_config)
        os.environ["MLPROJECT_MAIN_PROCESS"] = "1"

    # check the number of trials to repeat the experiment of a config
    if "nb_trial" not in exp_config:
        exp_config["nb_trial"] = 1

    if nb_consumer > 1 and exp_config["nb_trial"] > 1:
        raise RuntimeError("Distributed mode doesnt support multiple trials")

    # repeat the experiments of a configuration
    for trial_index in range(exp_config["nb_trial"]):
        # create a copy of the original config for each trial
        config = copy.deepcopy(exp_config)

        # assign trial index
        config["trial_index"] = trial_index
        config["nb_consumer"] = nb_consumer

        # prepare directories
        prepare_directories(config)

        # -------- DATA ---------------------------------------
        if config["data_loader_type"] == "torch":
            train_loader = get_torch_loader(config, "train")
            val_loader = get_torch_loader(config, "val")
            test_loader = get_torch_loader(config, "test")
        elif config["data_loader_type"] == "swift":
            train_loader = get_swift_loader(config, "train")
            val_loader = get_swift_loader(config, "val")
            test_loader = get_swift_loader(config, "test")
        else:
            raise ValueError(
                f"data loader {config['data_loader_type']} type is not supported"
            )

        try:
            # -------------- MODEL --------------------------------
            # TODO: create model here
            model = DenseNet(**config["model_configs"])
            # -----------------------------------------------------

            # create tensorboard logger here
            tensorboard_logger = None
            # prefix for this experiment
            logger_prefix = ""

            # create a trainer class
            trainer = get_trainer(config, device)
            trainer.fit(
                model,
                {"dataloader": train_loader},
                {"dataloader": val_loader},
                {"dataloader": test_loader},
                tensorboard_logger=tensorboard_logger,
                logger_prefix=logger_prefix,
                load_best=config[
                    "load_best"
                ],  # if True, will attempt to load the best checkpoint based on train/val perf
            )
        except BaseException as error:
            dispose_data_loader(train_loader, val_loader, test_loader)
            traceback.print_exc()
            raise error
        else:
            # clean up
            dispose_data_loader(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    args = parse_args()

    if "MLPROJECT_DEVICE_COUNT" in os.environ:
        nb_consumer = int(os.environ["MLPROJECT_DEVICE_COUNT"])
    else:
        nb_consumer = max(1, torch.cuda.device_count())
        os.environ["MLPROJECT_DEVICE_COUNT"] = str(nb_consumer)
        if nb_consumer == 1 and torch.cuda.is_available():
            print("RUNNING WITH 1 GPU")
        elif nb_consumer > 1:
            print(f"RUNNING WITH {nb_consumer} GPUs")

    # load config file
    config_values = load_config(args.config_path, args.index, args.test_mode)
    main(
        config_values,
        device=args.device,
        nb_consumer=nb_consumer,
        cli_kwargs=args.kwargs,
    )
