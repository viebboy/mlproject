"""
cli.py: command line interface for this package
-----------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-12-06
* Version: 0.0.1


This is part of the ml package


License
-------
Proprietary License

"""

from __future__ import annotations
import argparse
from model_training.cli_handlers import write_template_handler, train_handler


COMMAND_TO_HANDLER = {
    "generate-template": write_template_handler,
    "train": train_handler,
}


def parse_args():
    # Create the main parser
    parser = argparse.ArgumentParser(description="ml cli")

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Sub-commands to run with ml",
    )

    add_write_template_parser(subparsers)
    add_train_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()
    return args


def add_write_template_parser(subparsers):
    parser = subparsers.add_parser("generate-template", help="generate template files")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to directory to write",
    )


def add_train_parser(subparsers):
    parser = subparsers.add_parser("train", help="train model")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="config index",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="device to train on",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="if turn on, will run in test mode",
    )


def main():
    args = parse_args()
    if args.subcommand in COMMAND_TO_HANDLER:
        COMMAND_TO_HANDLER[args.subcommand](args)
    else:
        commands = "\n".join(list(COMMAND_TO_HANDLER.keys()))
        msg = (
            f"Unsupported subcommand: {args.subcommand}.\n"
            "It seems handler for this command has not been added\n"
            f"ml supports the following subcommands:\n"
            f"{commands}"
        )
        raise ValueError(msg)


if __name__ == "__main__":
    main()
