"""
train_entry.py: main training orchestration
-------------------------------------------


* Copyright: 2023 datsbot.com
* Authors: Dat Tran (business@datsbot.com)
* Date: 2023-12-22
* Version: 0.0.1

This is part of the ml project

License
-------
Proprietary License

"""

from __future__ import annotations
import os
import copy
import traceback
from model_training.tools import (
    get_data_loader,
    dispose_data_loader,
    get_trainer,
    get_checkpoint_callback,
    get_model,
    prepare_directories,
    print_config,
)


def main(
    config: dict,
    device: str,
    nb_consumer: int,
    cli_kwargs: dict = {},
) -> None:
    # overwrite values from cli kwargs
    for key, value in cli_kwargs.items():
        config[key] = value

    # print config
    if "MLPROJECT_MAIN_PROCESS" not in os.environ:
        print_config(config)
        os.environ["MLPROJECT_MAIN_PROCESS"] = "True"

    # assign trial index
    config["dataloader"]["nb_consumer"] = nb_consumer

    # prepare directories
    prepare_directories(config)

    # -------- DATA ---------------------------------------
    train_loader = get_data_loader(config, "train")
    val_loader = get_data_loader(config, "val")
    test_loader = get_data_loader(config, "test")
    checkpoint_callback = get_checkpoint_callback(config)
    if checkpoint_callback is not None:
        checkpoint_callback = {
            "constructor": checkpoint_callback[0],
            "arguments": checkpoint_callback[1]
            }
    try:
        # -------------- MODEL --------------------------------
        model = get_model(config)
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
            load_best=config["load_best"],
            checkpoint_callback=checkpoint_callback,
        )

    except KeyboardInterrupt:
        dispose_data_loader(train_loader, val_loader, test_loader)
        print("Training interrupted by user. Exit now...")
    except BaseException as error:
        dispose_data_loader(train_loader, val_loader, test_loader)
        traceback.print_exc()
        raise error
    else:
        # clean up
        dispose_data_loader(train_loader, val_loader, test_loader)
