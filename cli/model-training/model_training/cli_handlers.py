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
import optuna
from model_training.train_entry import main as train_entry_main
from model_training.tools import (
    load_config,
    get_config_size,
    load_module,
)


def write_classification_template_handler(args: argparse.Namespace) -> None:
    os.makedirs(args.path, exist_ok=True)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(current_dir, "template", "classification")
    config_file = os.path.join(template_dir, "config.py")
    dataset_file = os.path.join(template_dir, "dataset.py")
    loss_file = os.path.join(template_dir, "loss.py")
    metric_file = os.path.join(template_dir, "metric.py")
    model_file = os.path.join(template_dir, "model.py")
    gitignore_file = os.path.join(template_dir, "gitignore.txt")

    shutil.copy(config_file, os.path.join(args.path, "config.py"))
    shutil.copy(dataset_file, os.path.join(args.path, "dataset.py"))
    shutil.copy(loss_file, os.path.join(args.path, "loss.py"))
    shutil.copy(metric_file, os.path.join(args.path, "metric.py"))
    shutil.copy(model_file, os.path.join(args.path, "model.py"))
    shutil.copy(gitignore_file, os.path.join(args.path, ".gitignore"))


def write_regression_template_handler(args: argparse.Namespace) -> None:
    os.makedirs(args.path, exist_ok=True)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(current_dir, "template", "regression")
    config_file = os.path.join(template_dir, "config.py")
    dataset_file = os.path.join(template_dir, "dataset.py")
    loss_file = os.path.join(template_dir, "loss.py")
    metric_file = os.path.join(template_dir, "metric.py")
    model_file = os.path.join(template_dir, "model.py")
    gitignore_file = os.path.join(template_dir, "gitignore.txt")

    shutil.copy(config_file, os.path.join(args.path, "config.py"))
    shutil.copy(dataset_file, os.path.join(args.path, "dataset.py"))
    shutil.copy(loss_file, os.path.join(args.path, "loss.py"))
    shutil.copy(metric_file, os.path.join(args.path, "metric.py"))
    shutil.copy(model_file, os.path.join(args.path, "model.py"))
    shutil.copy(gitignore_file, os.path.join(args.path, ".gitignore"))


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
        cli_kwargs={},
    )


def train_handler(args: argparse.Namespace) -> None:
    nb_config = get_config_size(args.config_path)
    if args.index is not None:
        assert 0 <= args.index < nb_config, f"index must be in [0, {nb_config})"
        train_single_config_handler(args)
    else:
        config_path = os.path.abspath(args.config_path)
        # loop through index and launch model-training command with --index
        for index in range(nb_config):
            cmd = [
                "model-training",
                "train",
                "--config-path",
                config_path,
                "--index",
                str(index),
                "--device",
                args.device,
            ]
            if args.test_mode:
                cmd.append("--test-mode")
            status = os.system(" ".join(cmd))
            # check if the command failed
            if status != 0:
                raise RuntimeError(f"Failed to run with index {index}")


def search_handler(args: argparse.Namespace) -> None:
    """Handler for search command using Optuna for hyperparameter optimization.

    Uses Optuna to optimize hyperparameters by generating config files,
    running training, and evaluating results.
    """
    # Load the metaconfig module
    metaconfig_path = os.path.abspath(args.metaconfig_path)
    construct_meta_config = load_module(metaconfig_path, "construct_meta_config")
    meta_config = construct_meta_config()

    # Create Optuna study with SQLite storage
    storage = f"sqlite:///{meta_config.db_path()}"
    study = optuna.create_study(
        study_name="model_search",
        storage=storage,
        load_if_exists=True,
        direction="maximize",  # Assuming maximize score
    )

    def objective(trial: optuna.trial.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Generate config file for this trial
        # Returns both config_path and output_dir
        config_path, output_dir = meta_config.generate_config_file(trial)
        config_path = os.path.abspath(config_path)

        # Get the number of experiment indices (e.g., cross-validation folds)
        nb_experiment = get_config_size(config_path)

        # Run training for all experiment indices
        for index in range(nb_experiment):
            cmd = [
                "model-training",
                "train",
                "--config-path",
                config_path,
                "--index",
                str(index),
                "--device",
                args.device,
            ]
            if args.test_mode:
                cmd.append("--test-mode")

            status = os.system(" ".join(cmd))
            if status != 0:
                raise RuntimeError(
                    f"Training failed for trial {trial.number}, " f"index {index}"
                )

        # Get score from the meta_config
        # MetaConfig handles loading all experiment results and calculating
        # average score if there are multiple indices
        score = meta_config.get_score(output_dir)
        return score

    # Run optimization
    study.optimize(
        objective,
        n_trials=meta_config.nb_trials,
        n_jobs=meta_config.nb_parallel_trials,
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
