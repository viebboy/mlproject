#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r ${ROOT_DIR}/requirements.txt
pip install ${ROOT_DIR}/

# install model-training cli
pip install -r https://raw.githubusercontent.com/viebboy/swift-loader/main/requirements.txt
pip install git+https://github.com/viebboy/swift-loader.git

pip install -r $ROOT_DIR/cli/model-training/requirements.txt
pip install $ROOT_DIR/cli/model-training/
