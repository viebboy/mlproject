"""
trainer.py: custom trainer goes here


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-02-13
* Version: 0.0.1

This is part of the cifar10 example project

License
-------
Apache 2.0 License

"""

import os
import shutil
import dill
from mlproject.metric import (
    MSE,
    MAE,
    Accuracy,
    CrossEntropy,
    F1,
    Precision,
    Recall,
    MetricFromLoss,
    NestedMetric,
)
from mlproject.loss import (
    compose_losses_to_callable,
    get_CrossEntropy,
    get_MSE,
    get_MAE,
    CrossEntropy,
    MSE,
    MAE,
)
from mlproject.trainer import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler,
    Trainer as BaseTrainer,
)


# because we dont need to modify any logic in trainer, we simply remove the
# custom trainer code generated from the template
# we only need to fillin the get_trainer() function

def get_trainer(config_file: str, config: dict, config_name: str, config_index: int):
    """
    Returns trainer object
    parameters:
        config (dict): a dictionary that contains all configurations for constructing a trainer obj
    returns:
        a trainer instance
    """

    # prep output dir and log dir
    # note this is the metadir for all experiments
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])

    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])

    # need to create subdir as output dir and log dir for this particular exp
    config_name = config_name.replace(' ', '_')
    output_dir = os.path.join(config['output_dir'], config_name, '{:09d}'.format(config_index))
    log_dir = os.path.join(config['log_dir'], config_name, '{:09d}'.format(config_index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # now copy the original config file, the config index and the content of
    # the config to log and output dir for reference purposes
    shutil.copy(config_file, os.path.join(log_dir, os.path.basename(config_file)))
    shutil.copy(config_file, os.path.join(output_dir, os.path.basename(config_file)))
    with open(os.path.join(log_dir, f'config_index_{config_index}.dill'), 'wb') as fid:
        dill.dump(config, fid)
    with open(os.path.join(output_dir, f'config_index_{config_index}.dill'), 'wb') as fid:
        dill.dump(config, fid)

    # ---------------------------------------------------------

    # create lr scheduler
    if config['lr_scheduler'] == 'cosine':
        lr_scheduler = get_cosine_lr_scheduler(config['start_lr'], config['stop_lr'])
    elif config['lr_scheduler'] == 'multiplicative':
        lr_scheduler = get_multiplicative_lr_scheduler(
            config['start_lr'],
            config['epochs_to_drop_lr'], #the epoch number to drop learning rate
            config['lr_multiplicative_factor'], #factor to multiply to the learning rate to change
        )
    else:
        raise NotImplemented

    # we use the standard trainer implementation from the library
    trainer =  BaseTrainer(
        n_epoch=config['n_epoch'],
        output_dir=output_dir, #use directory created above
        loss_function=config['loss_function'],
        metrics=config['metrics'],
        monitor_metric=config['monitor_metric'],
        monitor_direction=config['monitor_direction'],
        checkpoint_idx=config['checkpoint_idx'],
        lr_scheduler=lr_scheduler,
        optimizer=config['optimizer'],
        weight_decay=config['weight_decay'],
        log_dir=log_dir, # use directory created above for this configuration
        checkpoint_freq=config['checkpoint_freq'],
        print_freq=config['print_freq'],
        eval_freq=config['eval_freq'],
        max_checkpoint=config['max_checkpoint'],
        use_progress_bar=config['use_progress_bar'],
        test_mode=config['test_mode'],
        move_data_to_device=config['move_data_to_device'],
        retain_metric_objects=config['retain_metric_objects'],
    )
    return trainer
