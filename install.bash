#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r ${ROOT_DIR}/requirements.txt
pip install -e .

# install model-training cli
pip install -r $ROOT_DIR/cli/model-training/requirements.txt
pip install -e $ROOT_DIR/cli/model-training/
