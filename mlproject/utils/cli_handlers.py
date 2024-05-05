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
import os
from mlproject.data import BinaryImageBlob


def create_image_blob_handler(args: argparse.Namespace) -> None:
    """handler for create-image-blob command"""

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Directory {args.image_dir} not found")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    blob = BinaryImageBlob(
        image_dir=args.image_dir,
        binary_file_dir=args.output_dir,
        nb_shard=args.nb_shard,
        save_decode=args.save_decode,
    )
