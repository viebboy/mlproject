"""
constants.py: all contants go here
----------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0


"""

from __future__ import annotations
import os
import sys
import json
from loguru import logger

# root dir is for user configuration
ROOT_DIR = os.path.join(os.path.expanduser('~'), '.mlproject')

# package dir is the path to the package source code
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


## ----- Handle Configuration and Initialization -------------------
if not os.path.exists(ROOT_DIR):
    logger.warning('mlproject is not initialized yet')
    logger.warning('Please run "mlproject init" to initialize before using')
    logger.warning('For complete options, run "mlproject --help"')
    logger.warning('If you are running "mlproject init", you can ignore this warning')
    config_file = os.path.join(PACKAGE_DIR, '.configuration.json')
else:
    config_file = os.path.join(ROOT_DIR, 'configurations.json')
    if not os.path.exists(config_file):
        logger.warning(f'cannot find configuration file with path {config_file}')
        logger.warning('Please run "mlproject init" to re-initialize mlproject')

with open(config_file, 'r') as fid:
    config = json.loads(fid.read())

# list of authors, specified in dictionaries
AUTHORS = config['AUTHORS']
# name of company
COMPANY = config['COMPANY']

# licenses
LICENSES = config['LICENSES']
DEFAULT_LICENSE = config['DEFAULT_LICENSE']

# disable warning, number of parallel jobs and log level can be overwritten
# with env vars
DISABLE_WARNING = config['DISABLE_WARNING']
if 'MLPROJECT_DISABLE_WARNING' in os.environ:
    if os.environ['MLPROJECT_DISABLE_WARNING'] in ['TRUE', 'true', 'True']:
        DISABLE_WARNING = True
    else:
        DISABLE_WARNING = False

NB_PARALLEL_JOBS = config['NB_PARALLEL_JOBS']
if 'MLPROJECT_NB_PARALLEL_JOBS' in os.environ:
    try:
        NB_PARALLEL_JOBS = int(os.environ['MLPROJECT_NB_PARALLEL_JOBS'])
    except Exception:
        pass

LOG_LEVEL = config['LOG_LEVEL']
if 'MLPROJECT_LOG_LEVEL' in os.environ:
    if os.environ['MLPROJECT_LOG_LEVEL'] in ['INFO', 'DEBUG']:
        LOG_LEVEL = os.environ['MLPROJECT_LOG_LEVEL']
