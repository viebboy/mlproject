"""
exp_summarizer.py: utility to generate a table of results for experiments
-------------------------------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-03-08
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0


"""

import os
import sys
import time
from loguru import logger
import importlib.util
import numpy as np
import dill
import pandas as pd


def load_config_module(file: str):
    """
    get the total number of configurations
    """
    try:
        spec = importlib.util.spec_from_file_location('config_module', file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    except Exception as error:
        logger.warning(f'faced error when trying to load the config file')
        raise error


def load_config_value(folder):
    filename = None
    for f in os.listdir(folder):
        if f.startswith('config_index_') and f.endswith('.dill'):
            filename = f
            break
    if filename is None:
        raise RuntimeError(f'Cannot find configuration value in {folder}')

    with open(os.path.join(folder, filename), 'rb') as fid:
        config_value = dill.load(fid)

    return config_value


def summarize_experiments(
    config_path: str,
    entry_script: str,
    output_file: str,
    delimiter,
    selected_metrics,
):
    if config_path is None:
        raise RuntimeError('path to configuration file must be provided via --config-path')
    if entry_script is None:
        raise RuntimeError('path to entry script must be provided via --entry-script')
    if selected_metrics is not None:
        selected_metrics = selected_metrics.split(',')

    entry_script = os.path.abspath(entry_script)
    config_script = os.path.abspath(config_path)
    root_dir = os.path.dirname(entry_script)
    sys.path.append(root_dir)

    # load the configuration module
    config_module = load_config_module(config_path)

    # create all configs and get the name
    all_configs = config_module.create_all_config()
    config_name = config_module.get_name()

    if not hasattr(config_module, 'ALL_CONFIGS'):
        raise RuntimeError('The configuration file doesnt contain "ALL_CONFIGS" that holds all configurations')

    # check output_dir existence
    if 'output_dir' not in all_configs[0]:
        msg = (
            'Cannot find key: "output_dir" in the configurations; ',
            '"output_dir" is required to parse the outputs from all configurations',
        )
        raise RuntimeError(''.join(msg))

    root = all_configs[0]['output_dir']
    output_dir = os.path.join(root, config_name.replace(' ', '_'))
    if not os.path.exists(output_dir):
        msg = (
            f'Cannot find the output_dir where all results should be saved at: {output_dir}; '
            f'The output_dir in configuration file is: {root}; ',
            f'The configuration name specified in configuration file is: {config_name}'
        )
        raise RuntimeError(''.join(msg))

    if output_file is None:
        output_file = os.path.join(output_dir, 'summary')

    # parse results from each experiment run
    result_dirs = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    result_dirs = sorted([f for f in result_dirs if os.path.isdir(f)])
    if len(result_dirs) == 0:
        raise RuntimeError(f'Cannot find any results under {output_dir}')

    results = []
    config_values = []
    for result_folder in result_dirs:
        # result_folder holds results of all trial runs for one experiment config
        trial_run_results = sorted([os.path.join(result_folder, f) for f in os.listdir(result_folder)])
        for folder in trial_run_results:
            if not os.path.isdir(folder):
                raise RuntimeError(
                    'Under the folder of an experiment config must be folders of different trial runs'
                )

        # load the config values
        config_values.append(load_config_value(trial_run_results[0]))

        # loop through the trial run and compute the mean and std result
        performance = {}
        for folder in trial_run_results:
            performance_file = os.path.join(folder, 'final_performance.dill')
            with open(performance_file, 'rb') as fid:
                p = dill.load(fid)
                metrics = p['train'].keys()
                for m in metrics:
                    key = 'train_' + m
                    if key not in performance:
                        try:
                            # try to access value of metric object
                            performance[key] = [p['train'][m].value()]
                        except:
                            # otherwise just append
                            performance[key] = [p['train'][m]]
                    else:
                        try:
                            performance[key].append(p['train'][m].value())
                        except:
                            performance[key].append(p['train'][m])

                    if m in p['val']:
                        key = 'val_' + m
                        if key not in performance:
                            try:
                                performance[key] = [p['val'][m].value()]
                            except:
                                performance[key] = [p['val'][m]]
                        else:
                            try:
                                performance[key].append(p['val'][m].value())
                            except:
                                performance[key].append(p['val'][m])

                    if m in p['test']:
                        key = 'test_' + m
                        if key not in performance:
                            try:
                                performance[key] = [p['test'][m].value()]
                            except:
                                performance[key] = [p['test'][m]]
                        else:
                            try:
                                performance[key].append(p['test'][m].value())
                            except:
                                performance[key].append(p['test'][m])

        if selected_metrics is None:
            metrics = list(performance.keys())
        else:
            all_metrics = list(performance.keys())
            metrics = []
            for m in selected_metrics:
                for v in all_metrics:
                    if m in v:
                        metrics.append(v)
            if len(metrics) == 0:
                logger.warning('--metrics was specified but the artifacts contain no such metric')
                logger.warning(f'user specified the following metrics: {selected_metrics}')
                logger.warning(f'artifacts contain the following metrics: {all_metrics}')
                raise RuntimeError('--metrics was specified but the artifacts contain no such metric')

        metrics.sort()
        summary = {}
        sorted_metrics = []
        for metric in metrics:
            if len(performance[metric]) > 1:
                summary[metric + ' (mean) '] = np.mean(performance[metric])
                summary[metric + ' (std) '] = np.std(performance[metric])
                sorted_metrics.append(metric + ' (mean) ')
                sorted_metrics.append(metric + ' (std) ')
            else:
                summary[metric] = performance[metric][0]
                sorted_metrics.append(metric)

        results.append(summary)

    # now we have all results from all experiment configs
    # let's put them in a table
    # to do so, we need to find which parameters are shared between all configs
    # and which ones are different
    ALL_CONFIGS = config_module.ALL_CONFIGS
    shared_parameters = []
    parameters = []
    for key in ALL_CONFIGS:
        # if this parameter only receives 1 value
        if len(ALL_CONFIGS[key]) == 1:
            shared_parameters.append(key)
        else:
            parameters.append(key)

    if delimiter is None:
        delimiter = '|'

    # dump a table of shared parameters
    shared_table = {p: [] for p in shared_parameters}
    with open(output_file + '_shared_parameters.csv', 'w') as fid:
        for p in shared_parameters:
            fid.write('{}{}{}\n'.format(p, delimiter, str(list(ALL_CONFIGS[p])[0])))
            shared_table[p].append(str(list(ALL_CONFIGS[p])[0]))

    with open(output_file + '_shared_parameters.html', 'w') as fid:
        shared_table = pd.DataFrame.from_dict(shared_table)
        shared_table.index = ['Parameter Value',]
        fid.write(shared_table.transpose().to_html())

    # then dump parameters with values
    headers = sorted_metrics + parameters
    table = {h: [] for h in headers}
    for config, result in zip(config_values, results):
        for key in sorted_metrics:
            table[key].append(result[key])
        for key in parameters:
            table[key].append(str(config[key]))

    table = pd.DataFrame.from_dict(table)
    table.index = [f'config_index={i}' for i in range(len(results))]
    table.to_csv(output_file + '_results.csv', sep=delimiter)
    with open(output_file + '_results.html', 'w') as fid:
        fid.write(table.to_html())
