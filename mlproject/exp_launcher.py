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
import GPUtil


def thread_worker(
    entry_file,
    config_file,
    device,
    input_queue,
    output_queue,
    close_event
):
    while True:
        if close_event.is_set():
            logger.info('receive information to close the worker thread, closing now...')
            break

        if not input_queue.empty():
            config_index, gpu_indices = input_queue.get()
            env_var = os.environ.copy()
            if gpu_indices is not None:
                env_var['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_indices)

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
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            process.wait()
            if process.returncode != 0:
                msg = (
                    f'the returncode ({process.returncode}) from this run (index={config_index}) is non-zero, ',
                    'terminating now...'
                )
                logger.warning(''.join(msg))
            output_queue.put(
                {
                    'returncode': process.returncode,
                    'stderr': process.stderr.read().decode(),
                    'gpu_indices': gpu_indices,
                    'config_index': config_index,
                }
            )
        else:
            time.sleep(1)


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


def launch_on_gpu(entry_file, config_file, nb_parallel_exp, gpu_indices, gpu_per_exp):
    total = get_nb_of_total_exp(config_file)
    config_indices = list(range(total))
    indices_being_run = []
    indices_completed = []

    nb_threads = min(nb_parallel_exp, total, int(len(gpu_indices) / gpu_per_exp))
    input_queues = [Queue() for _ in range(nb_threads)]
    output_queues = [Queue() for _ in range(nb_threads)]
    close_events = [threading.Event() for _ in range(nb_threads)]

    threads = [
        threading.Thread(
            target=thread_worker,
            args=(
                entry_file,
                config_file,
                'gpu',
                input_queue,
                output_queue,
                close_event
            )
        ) for input_queue, output_queue, close_event in zip(input_queues, output_queues, close_events)
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
                len(gpu_indices) > gpu_per_exp
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



def launch_on_cpu(entry_file, config_file, nb_parallel_exp):
    total = get_nb_of_total_exp(config_file)
    config_indices = list(range(total))
    indices_being_run = []
    indices_completed = []

    nb_threads = min(nb_parallel_exp, total)
    input_queues = [Queue() for _ in range(nb_threads)]
    output_queues = [Queue() for _ in range(nb_threads)]
    close_events = [threading.Event() for _ in range(nb_threads)]

    threads = [
        threading.Thread(
            target=thread_worker,
            args=(
                entry_file,
                config_file,
                'cpu',
                input_queue,
                output_queue,
                close_event
            )
        ) for input_queue, output_queue, close_event in zip(input_queues, output_queues, close_events)
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


def exp_launcher(entry_file, config_file, device, gpu_indices, gpu_per_exp, nb_parallel_exp):
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
        all_indices = GPUtil.getAvailable()
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

        if len(gpu_indices) % gpu_per_run != 0:
            msg = (
                f'there are {len(gpu_indices)} GPU(s) but each exp run requires {gpu_per_run} GPU(s) ',
                'so we cannot utilize all GPUs at the same time'
            )
            logger.warning(''.join(msg))

        launch_on_gpu(entry_file, config_file, nb_parallel_exp, gpu_indices, gpu_per_exp)
    else:
        launch_on_cpu(entry_file, config_file, nb_parallel_exp)
