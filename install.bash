#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r ${ROOT_DIR}/requirements.txt
pip install -e .
