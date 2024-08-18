#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r ${ROOT_DIR}/requirements.txt
pip install ${ROOT_DIR}/

# install model-training cli
pip install -r $ROOT_DIR/cli/model-training/requirements.txt
pip install $ROOT_DIR/cli/model-training/
