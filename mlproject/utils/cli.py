"""
util_cli.py: command line interface for this package
----------------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-12-06
* Version: 0.0.1


This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0

"""

from __future__ import annotations
import argparse
from mlproject.utils.cli_handlers import create_image_blob_handler

COMMAND_TO_HANDLER = {
    "create-image-blob": create_image_blob_handler,
}


def parse_args():
    # Create the main parser
    parser = argparse.ArgumentParser(description="mlproject-utils cli")

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Sub-commands to run with mypackage",
    )

    add_create_image_blob_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()
    return args


def add_create_image_blob_parser(subparsers):
    parser = subparsers.add_parser(
        "create-image-blob", help="create binary image blob from image dir"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="path to image dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="path to output dir that contains binary files",
    )
    parser.add_argument(
        "--nb-shard",
        type=int,
        required=True,
        help="number of binary shards",
    )
    # bool flag
    parser.add_argument(
        "--save-decode",
        action="store_true",
        help="whether to decode images and save them in decoded form",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="whether to benchmark the blob",
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
            f"mlproject-utils supports the following subcommands:\n"
            f"{commands}"
        )
        raise ValueError(msg)


if __name__ == "__main__":
    main()
