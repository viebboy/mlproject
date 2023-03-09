"""
exp_launcher.py: experiment launcher utility
--------------------------------------------


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
import threading
import time
import subprocess
from loguru import logger
from queue import Queue
import importlib.util
import torch


def thread_worker(
    entry_file,
    config_file,
    device,
    input_queue,
    output_queue,
    close_event,
    log_prefix
):
    process = None
    if log_prefix is not None:
        stderr = open(log_prefix + '.err', 'a')
        stdout = open(log_prefix + '.out', 'a')
    else:
        stderr = subprocess.PIPE
        stdout = subprocess.PIPE

    while True:
        if close_event.is_set():
            logger.info('receive information to close the worker thread, closing now...')
            if process is not None:
                process.kill()
                process.wait()
            break

        if not input_queue.empty():
            config_index, gpu_indices = input_queue.get()
            env_var = os.environ.copy()
            if gpu_indices is not None:
                env_var['CUDA_VISIBLE_DEVICES'] = ','.join([str(v) for v in gpu_indices])

            if device == 'cuda':
                msg = (
                    f'running config index {config_index} on ',
                    '{}'.format(', '.join([f'GPU{i}' for i in gpu_indices]))
                )
                logger.info(''.join(msg))
            else:
                logger.info(f'running config index {config_index} on CPU')

            process = subprocess.Popen(
                [
                    'python',
                    entry_file,
                    '--config-path',
                    config_file,
                    '--index',
                    str(config_index),
                    '--device',
                    device,
                ],
                env=env_var,
                stderr=stderr,
                stdout=stdout,
            )
            is_terminated = False
            while True:
                # now wait for the process to finish or signal to kill
                if close_event.is_set():
                    process.kill()
                    process.wait()
                    is_terminated = True
                    break

                # if not killed, then poll
                returncode = process.poll()
                if returncode is None:
                    time.sleep(5)
                else:
                    if process.returncode != 0:
                        msg = (
                            f'the returncode ({process.returncode}) from this run (index={config_index}) is non-zero, ',
                            'terminating now...'
                        )
                        logger.warning(''.join(msg))

                    if log_prefix is not None:
                        err_msg = (
                            'Error occurred in worker. Because --worker-log-prefix was provided, ',
                            'error message has been redirected to ',
                            '{}'.format(log_prefix + '.err')
                        )
                        err_msg = ''.join(err_msg)
                    else:
                        err_msg = process.stderr.read().decode()

                    output_queue.put(
                        {
                            'returncode': process.returncode,
                            'stderr': err_msg,
                            'gpu_indices': gpu_indices,
                            'config_index': config_index,
                        }
                    )
                    break

            if is_terminated:
                break
        else:
            time.sleep(1)

    if log_prefix is not None:
        stdout.close()
        stderr.close()


def get_nb_of_total_exp(file: str):
    """
    get the total number of configurations
    """
    try:
        spec = importlib.util.spec_from_file_location('config_module', file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        configs = config_module.create_all_config()
        return len(configs)
    except Exception as error:
        logger.warning(f'faced error when trying to load the config file')
        raise error


def launch_on_gpu(entry_file, config_file, nb_parallel_exp, gpu_indices, gpu_per_exp, log_prefix):
    total = get_nb_of_total_exp(config_file)
    logger.info(f'total number of configuration indices: {total}')
    config_indices = list(range(total))
    indices_being_run = []
    indices_completed = []

    nb_threads = min(nb_parallel_exp, total, int(len(gpu_indices) / gpu_per_exp))
    input_queues = [Queue() for _ in range(nb_threads)]
    output_queues = [Queue() for _ in range(nb_threads)]
    close_events = [threading.Event() for _ in range(nb_threads)]

    logger.info(f'running a maximum of {nb_threads} workers in parallel on GPU')

    if log_prefix is None:
        log_files = [None for _ in range(nb_threads)]
    else:
        log_files = [log_prefix + f'_worker_{i}' for i in range(nb_threads)]

    threads = [
        threading.Thread(
            target=thread_worker,
            args=(
                entry_file,
                config_file,
                'cuda',
                input_queue,
                output_queue,
                close_event,
                log_file,
            )
        ) for input_queue, output_queue, close_event, log_file in zip(
            input_queues,
            output_queues,
            close_events,
            log_files,
        )
    ]
    for thread in threads:
        thread.start()

    running_mask = [False for _ in range(nb_threads)]

    try:
        while len(indices_completed) < total:
            # launch a run
            new_launch = (
                len(indices_being_run) < nb_parallel_exp and
                sum(running_mask) < len(running_mask) and
                len(config_indices) > 0 and
                len(gpu_indices) >= gpu_per_exp
            )
            if new_launch:
                for i in range(nb_threads):
                    if not running_mask[i]:
                        index = config_indices.pop(0)
                        gpu_indices_ = [gpu_indices.pop(0) for _ in range(gpu_per_exp)]
                        running_mask[i] = True
                        input_queues[i].put((index, gpu_indices_))
                        indices_being_run.append(index)
                        break
            else:
                time.sleep(5)

            # check if anything complete
            for i in range(nb_threads):
                if not output_queues[i].empty():
                    result = output_queues[i].get()
                    if result['returncode'] != 0:
                        # have issue
                        msg = (
                            f'faced issue when running configuration index: {result["config_index"]}. ',
                            f'returncode is: {result["returncode"]}'
                            f'stderr is: {result["stderr"]}'
                        )
                        # clean up
                        raise RuntimeError(''.join(msg))

                    indices_completed.append(result['config_index'])
                    indices_being_run.remove(result['config_index'])
                    gpu_indices.extend(result['gpu_indices'])
                    running_mask[i] = False

                    logger.info(f'complete configuration index : {result["config_index"]}')
                    logger.info('completion percentage: {:.2f} %'.format(
                        len(indices_completed) * 100 / total
                    ))

    except Exception as error:
        for event in close_events:
            event.set()
        for thread in threads:
            thread.join()
        raise error
    else:
        for event in close_events:
            event.set()
        for thread in threads:
            thread.join()
        logger.info('done running all experiments')



def launch_on_cpu(entry_file, config_file, nb_parallel_exp, log_prefix):
    total = get_nb_of_total_exp(config_file)
    logger.info(f'total number of configuration indices: {total}')
    config_indices = list(range(total))
    indices_being_run = []
    indices_completed = []

    nb_threads = min(nb_parallel_exp, total)
    input_queues = [Queue() for _ in range(nb_threads)]
    output_queues = [Queue() for _ in range(nb_threads)]
    close_events = [threading.Event() for _ in range(nb_threads)]

    logger.info(f'running a maximum of {nb_threads} workers in parallel on CPU')

    if log_prefix is None:
        log_files = [None for _ in range(nb_threads)]
    else:
        log_files = [log_prefix + f'_worker_{i}' for i in range(nb_threads)]

    threads = [
        threading.Thread(
            target=thread_worker,
            args=(
                entry_file,
                config_file,
                'cpu',
                input_queue,
                output_queue,
                close_event,
                log_file,
            )
        ) for input_queue, output_queue, close_event, log_file in zip(
            input_queues,
            output_queues,
            close_events,
            log_files,
        )
    ]
    for thread in threads:
        thread.start()

    running_mask = [False for _ in range(nb_threads)]

    try:
        while len(indices_completed) < total:
            # launch a run
            new_launch = (
                len(indices_being_run) < nb_parallel_exp and
                sum(running_mask) < len(running_mask) and
                len(config_indices) > 0
            )
            if new_launch:
                for i in range(nb_threads):
                    if not running_mask[i]:
                        index = config_indices.pop(0)
                        running_mask[i] = True
                        input_queues[i].put((index, None))
                        indices_being_run.append(index)
                        break
            else:
                time.sleep(5)

            # check if anything complete
            for i in range(nb_threads):
                if not output_queues[i].empty():
                    result = output_queues[i].get()
                    if result['returncode'] != 0:
                        # have issue
                        msg = (
                            f'faced issue when running configuration index: {result["config_index"]}. ',
                            f'returncode is: {result["returncode"]}, '
                            f'stderr is: {result["stderr"]}'
                        )
                        raise RuntimeError(''.join(msg))

                    indices_completed.append(result['config_index'])
                    indices_being_run.remove(result['config_index'])
                    running_mask[i] = False
                    logger.info(f'complete configuration index : {result["config_index"]}')
                    logger.info('completion percentage: {:.2f} %'.format(
                        len(indices_completed) * 100 / total
                    ))

    except Exception as error:
        for event in close_events:
            event.set()
        for thread in threads:
            thread.join()
        raise error
    else:
        for event in close_events:
            event.set()
        for thread in threads:
            thread.join()
        logger.info('done running all experiments')


def exp_launcher(entry_file, config_file, device, gpu_indices, gpu_per_exp, nb_parallel_exp, log_prefix):
    if entry_file is None:
        raise RuntimeError('path to entry script must be specified via --entry-file')

    if config_file is None:
        raise RuntimeError('path to config file must be specified via --config-path')

    if device not in ['cuda', 'cpu']:
        raise RuntimeError('--device must be either "cuda" or "cpu"')

    if device == 'cuda':
        if gpu_per_exp is None:
            raise RuntimeError('the number of GPUs to use per experiment must be given via --gpu-per-exp')

    else:
        if nb_parallel_exp is None:
            msg = (
                'when training on CPU, the maximum number of parallel experiments ',
                'must be given via --nb-parallel-exp'
            )
            raise RuntimeError(''.join(msg))

    entry_file = os.path.abspath(entry_file)
    config_file = os.path.abspath(config_file)

    if device == 'cuda':
        all_indices = list(range(torch.cuda.device_count()))
        if len(all_indices) == 0:
            raise RuntimeError('the running device is CUDA but cannot detect any GPUs')

        if gpu_indices is None:
            gpu_indices = all_indices
        else:
            try:
                gpu_indices = [int(v) for v in gpu_indices.split(',')]
            except Exception as error:
                raise RuntimeError('--gpu-indices must be given as comma separated. For example 0,1,3')

            for idx in gpu_indices:
                assert idx in all_indices

        if len(gpu_indices) < gpu_per_exp:
            msg = (
                f'total number of GPUs ({len(gpu_indices)}) ',
                f'is less than gpu_per_exp ({gpu_per_exp})',
            )
            raise RuntimeError(''.join(msg))

        if len(gpu_indices) % gpu_per_exp != 0:
            msg = (
                f'there are {len(gpu_indices)} GPU(s) but each exp run requires {gpu_per_exp} GPU(s) ',
                'so we cannot utilize all GPUs at the same time'
            )
            logger.warning(''.join(msg))

        if nb_parallel_exp is not None:
            msg = (
                'users should not use --nb-parallel-exp when using cuda ',
                'the number of parallel experiments is determined via the number of GPUs specified ',
                'and the number of gpu per experiment specified by the users',
            )
            raise RuntimeError(''.join(msg))

        nb_parallel_exp = len(gpu_indices)

        launch_on_gpu(entry_file, config_file, nb_parallel_exp, gpu_indices, gpu_per_exp, log_prefix)
    else:
        launch_on_cpu(entry_file, config_file, nb_parallel_exp, log_prefix)
