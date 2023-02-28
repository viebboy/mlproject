"""
trainer.py: base trainer implementation for mlproject
-----------------------------------------------------


* Copyright: Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-01
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
from glob import glob
import pandas as pd
import plotly.express as px
import time
import numpy as np
import tempfile
from loguru import logger as guru_logger
import dill
from typing import Callable
import copy

from mlproject.metric import Metric


def get_cosine_lr_scheduler(init_lr, final_lr):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = final_lr + 0.5 * (init_lr - final_lr) * (1 + np.cos(np.pi * epoch_idx / n_epoch))
        return lr

    return lr_scheduler


def get_multiplicative_lr_scheduler(init_lr, drop_at, multiplicative_factor):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = init_lr
        for epoch in drop_at:
            if epoch_idx + 1 >= epoch:
                lr *= multiplicative_factor
        return lr

    return lr_scheduler


class Logger:
    def __init__(self, *loggers):
        self.loggers = loggers
        self.file_handles = []

    def add_file_handle(self, fid):
        self.file_handles.append(fid)

    def close(self):
        for fid in self.file_handles:
            fid.close()
        self.file_handles = []

    def timestamp(self):
        return str(pd.Timestamp.now()).split('.')[0]

    def info(self, msg):
        for logger in self.loggers:
            logger.info(msg)
        for fid in self.file_handles:
            fid.write(f'{self.timestamp()} INFO: {msg}\n')

    def debug(self, msg):
        for logger in self.loggers:
            logger.debug(msg)
        for fid in self.file_handles:
            fid.write(f'{self.timestamp()} DEBUG: {msg}\n')

    def warning(self, msg):
        for logger in self.loggers:
            logger.warning(msg)
        for fid in self.file_handles:
            fid.write(f'{self.timestamp()} WARN: {msg}\n')


class Trainer:
    n_test_minibatch = 100

    def __init__(
        self,
        n_epoch: int,
        output_dir: str,
        loss_function: Callable,
        metrics: list,
        monitor_metric: str,
        monitor_direction: str,
        checkpoint_idx: int=-1,
        lr_scheduler: Callable=get_cosine_lr_scheduler(1e-3, 1e-5),
        optimizer: str='adam',
        weight_decay: float=1e-4,
        log_dir: str=None,
        checkpoint_freq: int=10,
        max_checkpoint: int=10,
        eval_freq: int=1,
        print_freq: int=10,
        use_progress_bar: bool=True,
        test_mode: bool=False,
        move_data_to_device: bool=True,
        retain_metric_objects: bool=True,
        sample_input=None,
        logger=guru_logger,
    ):

        if not (isinstance(eval_freq, int) and eval_freq >= 1):
            msg = (
                'eval_freq must be an integer >= 1; ',
                f'received eval_freq={eval_freq}'
            )
            raise RuntimeError(''.join(msg))

        if not (isinstance(print_freq, int) and print_freq >= 1):
            msg = (
                'print_freq must be an integer >= 1; ',
                f'received print_freq={print_freq}'
            )
            raise RuntimeError(''.join(msg))

        self.output_dir = output_dir
        self.n_epoch = n_epoch
        self.epoch_idx = 0
        self.checkpoint_idx = checkpoint_idx
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.log_dir = log_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoint = max_checkpoint
        self.eval_freq = eval_freq
        self.print_freq = print_freq
        self.use_progress_bar = use_progress_bar
        self.test_mode = test_mode
        self.move_data_to_device = move_data_to_device
        self.retain_metric_objects = retain_metric_objects
        self.sample_input = sample_input
        # wrap user's logger into another interface that allows adding file
        # handling
        self.logger = Logger(logger)

        valid = False
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise RuntimeError(
                    f'the given metric {metric} must inherit from mlproject.metrics.Metric class'
                )
            if metric.name() == monitor_metric:
                valid = True

        if not valid:
            raise RuntimeError(f'monitor_metric={monitor_metric} does not exist in the list of `metrics`')

        self.history = {}

        self.loss_function = loss_function
        self.metrics = metrics
        self.monitor_metric = monitor_metric
        self.monitor_direction = monitor_direction

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.final_checkpoint_file = os.path.join(self.output_dir, 'final_checkpoint.pt')
        self.onnx_checkpoint_file = os.path.join(self.output_dir, 'final_checkpoint.onnx')
        self.final_performance_file = os.path.join(self.output_dir, 'final_performance.dill')

    def prepare_log_dir(self):

        if self.log_dir in ['', None]:
            self.log_dir_obj = tempfile.TemporaryDirectory()
            self.log_dir = self.log_dir_obj.name
        else:
            self.log_dir_obj = None

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def has_final_artifacts(self):
        has_pt_ckpt = os.path.exists(self.final_checkpoint_file)
        has_onnx_ckpt = os.path.exists(self.onnx_checkpoint_file)
        has_perf_file = os.path.exists(self.final_performance_file)
        if has_pt_ckpt and has_onnx_ckpt and has_perf_file:
            return True
        else:
            return False

    def _move_data_to_device(self, input_data, device, add_batch_axis=False):
        if isinstance(input_data, (list, tuple)):
            # there is a nested structure
            # create sub list for each item
            output_data = []
            for item in input_data:
                output_data.append(self._move_data_to_device(item, device, add_batch_axis))
            return output_data
        else:
            if add_batch_axis:
                return input_data.to(device).unsqueeze(0)
            else:
                return input_data.to(device)

    def verify_data(self, data: dict):
        if data is None:
            return
        msg = (
            'training/validation/test data must be passed as a dictionary that contains ',
            '"dataloader" as key',
        )
        if not isinstance(data, dict) or 'dataloader' not in data:
            raise RuntimeError(msg)

    def fit(
        self,
        model: torch.nn.Module,
        train_data: dict,
        val_data: dict = None,
        test_data: dict = None,
        device=torch.device('cpu'),
        tensorboard_logger=None,
        logger_prefix='',
    ):

        self.verify_data(train_data)
        self.verify_data(val_data)
        self.verify_data(test_data)

        if self.test_mode:
            self.total_minibatch = min(self.n_test_minibatch, len(train_data['dataloader']))
        else:
            self.total_minibatch = len(train_data['dataloader'])

        self.total_train_minibatch = self.n_epoch * self.total_minibatch

        # look for final checkpoint
        if self.has_final_artifacts():
            self.logger.warning(f'final artifacts exist under {self.output_dir}')
            self.logger.warning('no training was done')
            self.logger.warning(f'if you want to retrain model, please remove them')

            # load performance
            with open(self.final_performance_file, 'rb') as fid:
                performance = dill.load(fid)

            return performance


        self.prepare_log_dir()
        n_epoch_done = 0

        model.float()
        model.to(device, non_blocking=True)
        model.train()

        optimizer = self.get_optimizer(model)
        self.load_from_checkpoint(model, optimizer, device)
        self.start_time = time.time()
        self.start_minibatch_idx = self.cur_minibatch

        while self.epoch_idx < self.n_epoch:
            # optimize one epoch
            self.update_lr(optimizer)
            self.update_loop(model, train_data, optimizer, device)
            self.epoch_idx += 1

            if (self.epoch_idx % self.eval_freq) == 0:
                train_performance = self.eval(model, train_data, device, 'train set')
                val_performance = self.eval(model, val_data, device, 'val set')
                test_performance = self.eval(model, test_data, device, 'test set')
                self.update_metrics(train_performance, val_performance, test_performance)

                self.print_metrics(train_performance, 'train')
                self.print_metrics(val_performance, 'val')
                self.print_metrics(test_performance, 'test')

        # load the best model based on validation performance if exist, or train performance
        self.load_best(model)

        # eval this best model
        self.logger.info('evaluating performance of the final model...')
        final_train_metrics = self.eval(model, train_data, device, 'train set')
        final_val_metrics = self.eval(model, val_data, device, 'val set')
        final_test_metrics = self.eval(model, test_data, device, 'test set')

        self.print_metrics(final_train_metrics, 'final_train')
        self.print_metrics(final_val_metrics, 'final_val')
        self.print_metrics(final_test_metrics, 'final_test')

        # clean up temp dir
        if self.log_dir_obj is not None:
            self.log_dir_obj.cleanup()

        performance_curves = {}
        for key, value_list in self.history.items():
            if isinstance(value_list[0], (int, float)):
                performance_curves[key] = value_list
            else:
                performance_curves[key] = [metric_object.value() for metric_object in value_list]

        # close the file handle of loss curve
        self.loss_curve.close()
        self.visualize_performance(performance_curves)

        performance = {
            'history': self.history,
            'train': final_train_metrics,
            'val': final_val_metrics,
            'test': final_test_metrics,
        }

        # save performance
        with open(self.final_performance_file, 'wb') as fid:
            dill.dump(performance, fid, recurse=True)

        # save torch checkpoint
        model.cpu()
        torch.save(model, self.final_checkpoint_file)

        # save onnx checkpoint
        self.export_to_onnx(model, self.sample_input, self.onnx_checkpoint_file)

        if isinstance(self.logger, Logger):
            self.logger.close()

        return performance

    def export_to_onnx(self, model, sample_input, onnx_path):
        if sample_input is None:
            raise RuntimeError('sample_input is None; exporting a model to ONNX requires sample input')

        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={
                'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}
            }
        )
        self.logger.info(f'save model in ONNX format in {onnx_path}')

    def visualize_performance(self, performance):
        """
        visualize training curves
        """
        # now find the names of all metrics in performance
        metric_names = set(
            [x.replace('train_', '').replace('val_', '').replace('test_', '') for x in performance.keys()]
        )

        for metric in metric_names:
            plot_metrics = {'name': [], 'epoch': [], 'metric_value': []}
            x_ticks = self.history_indices
            nb_x_points = len(x_ticks)

            # train curve
            for idx in range(nb_x_points):
                plot_metrics['name'].append('train_' + metric)
                plot_metrics['epoch'].append(x_ticks[idx])
                plot_metrics['metric_value'].append(performance['train_' + metric][idx])

            # val curve
            if 'val_' + metric in performance:
                for idx in range(nb_x_points):
                    plot_metrics['name'].append('val_' + metric)
                    plot_metrics['epoch'].append(x_ticks[idx])
                    plot_metrics['metric_value'].append(performance['val_' + metric][idx])

            # test curve
            if 'test_' + metric in performance:
                for idx in range(nb_x_points):
                    plot_metrics['name'].append('test_' + metric)
                    plot_metrics['epoch'].append(x_ticks[idx])
                    plot_metrics['metric_value'].append(performance['test_' + metric][idx])


            df = pd.DataFrame(plot_metrics)
            fig = px.line(df, x = "epoch", y = "metric_value", color = "name", title = metric)
            filename = os.path.join(self.output_dir, f'{metric}.html')
            fig.write_html(filename)

        # plot performance curve
        loss_curve_file = os.path.join(self.output_dir, 'loss_curve.txt')
        with open(loss_curve_file, 'r') as fid:
            data = fid.read().split('\n')[:-1]
            loss_curve = []
            minibatch_indices = []
            for item in data:
                loss_curve.append(float(item.split(',')[1]))
                minibatch_indices.append(int(item.split(',')[0]))

        df = pd.DataFrame({'minibatch': minibatch_indices, 'loss value': loss_curve, 'name': 'train loss'})
        fig = px.line(df, x = "minibatch", y = "loss value", color = "name", title = 'train loss curve')
        filename = os.path.join(self.output_dir, 'loss_curve.html')
        fig.write_html(filename)

    def get_metric_value(self, metric):
        if isinstance(metric, (int, float)):
            return metric
        elif hasattr(metric, 'value') and callable(metric.value):
            return metric.value()
        else:
            raise RuntimeError('Cannot access metric value')

    def load_best(self, model):
        # load the best model from checkpoints based on monitor_metric
        has_val = 'val_' + self.monitor_metric in self.history

        if has_val and len(self.history['val_' + self.monitor_metric]) > 0:
            best_value = self.history['val_' + self.monitor_metric][-1]
            self.logger.info('loading the best checkpoint based on performance measured on validation data')
        else:
            best_value = self.history['train_' + self.monitor_metric][-1]
            self.logger.info('loading the best checkpoint based on performance measured on train data')

        best_value = self.get_metric_value(best_value)
        state_dict = model.state_dict()

        checkpoint_files = [
            os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith('checkpoint_')
        ]

        for filename in checkpoint_files:
            fid = open(filename, 'rb')
            checkpoint = dill.load(fid)
            fid.close()

            if has_val and len(checkpoint['history']['val_' + self.monitor_metric]) > 0:
                metric_value = checkpoint['history']['val_' + self.monitor_metric][-1]
            else:
                metric_value = checkpoint['history']['train_' + self.monitor_metric][-1]

            metric_value = self.get_metric_value(metric_value)

            if (self.monitor_direction == 'lower' and metric_value < best_value) or\
                    (self.monitor_direction == 'higher' and metric_value > best_value):
                best_value = metric_value
                state_dict = checkpoint['model_state_dict']

        model.load_state_dict(state_dict)

    def get_optimizer(self, model):
        assert self.optimizer in ['adam', 'sgd', 'adamW'],\
            'Given optimizer "{}" is not supported'.format(self.optimizer)

        # get current learning rate
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)

        # get separate batchnorm parameters and other parameters
        # if .get_parameters() is implemented in the model
        if hasattr(model, 'get_parameters') and callable(model.get_parameters):
            bn_params, other_params = model.get_parameters()

            if len(bn_params) > 0:
                params = [{'params': bn_params, 'weight_decay': 0},
                          {'params': other_params, 'weight_decay': self.weight_decay}]
            else:
                params = [{'params': other_params, 'weight_decay': self.weight_decay}]

            if self.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=lr)
            elif self.optimizer == 'sgd':
                optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
            elif self.optimizer == 'adamW':
                optimizer = optim.AdamW(params, lr=lr)
            else:
                raise NotImplemented
        else:
            if self.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), weight_decay=self.weight_decay, lr=lr)
            elif self.optimizer == 'sgd':
                optimizer = optim.SGD(model.parameters(),
                                      weight_decay=self.weight_decay,
                                      lr=lr,
                                      momentum=0.9,
                                      nesterov=True)
            elif self.optimizer == 'adamW':
                optimizer = optim.AdamW(model.parameters(), weight_decay=self.weight_decay, lr=lr)
            else:
                raise NotImplemented

        return optimizer

    def eval(self, model: torch.nn.Module, data: dict, device, dataset_name: str):
        if data is None or data['dataloader'] is None:
            return {}

        self.logger.info(f'evaluating {dataset_name}...')
        model.eval()
        # reset the metric objects
        for m in self.metrics:
            m.reset()

        # create fresh copy of metric objects
        metrics = copy.deepcopy(self.metrics)

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(data['dataloader']))
        else:
            total_minibatch = len(data['dataloader'])

        if self.use_progress_bar:
            loader = tqdm(data['dataloader'], desc=f'#Evaluating {dataset_name}: ', ncols=120, ascii=True)
        else:
            loader = data['dataloader']

        with torch.no_grad():
            for minibatch_idx, (inputs, labels) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                if self.move_data_to_device:
                    inputs = self._move_data_to_device(inputs, device)
                    labels = self._move_data_to_device(labels, device)

                predictions = model(inputs)
                for m in metrics:
                    m.update(predictions=predictions, labels=labels)

        # note here that we are collecting the metric object
        # not just the value
        performance = {}
        if self.retain_metric_objects:
            for m in metrics:
                performance[m.name()] = m
        else:
            for m in metrics:
                performance[m.name()] = m.value()

        return performance

    def update_lr(self,
                  optimizer):
        # update learning rate using lr_scheduler
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_loop(self, model, data, optimizer, device):
        total_time = 0
        forward_time = 0
        backward_time = 0
        data_prep_time = 0
        data_to_gpu_time = 0
        profile_counter = 0

        model.train()

        start_stamp = time.time()

        for inputs, labels in data['dataloader']:
            pre_gpu_stamp = time.time()

            if self.move_data_to_device:
                inputs = self._move_data_to_device(inputs, device)
                labels = self._move_data_to_device(labels, device)

            pre_forward_stamp = time.time()
            predictions = model(inputs)
            post_forward_stamp = time.time()
            loss = self.loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            backward_stamp = time.time()
            optimizer.zero_grad()

            # accumulate loss function
            self.accumulated_loss += loss.item()
            self.accumulated_loss_counter += 1

            # update information for profiling
            forward_time += (post_forward_stamp - pre_forward_stamp)
            backward_time += (backward_stamp - post_forward_stamp)
            total_time += (backward_stamp - start_stamp)
            data_prep_time += (pre_gpu_stamp - start_stamp)
            data_to_gpu_time += (pre_forward_stamp - pre_gpu_stamp)
            profile_counter += 1

            # increment counter for minibatch
            # minibatch_idx is index in the current epoch
            # cur_minibatch is index in the whole training loop
            self.minibatch_idx += 1
            self.cur_minibatch += 1

            # record the sample input for ONNX export if user doesnt provide
            if self.sample_input is None:
                self.sample_input = self._move_data_to_device(inputs, torch.device('cpu'))

            if (self.cur_minibatch % self.print_freq) == 0:
                # printing if needed
                self.print_and_update(
                    total_time,
                    data_prep_time,
                    data_to_gpu_time,
                    forward_time,
                    backward_time,
                    profile_counter,
                )
                # reset
                total_time = 0
                data_prep_time = 0
                data_to_gpu_time = 0
                forward_time = 0
                backward_time = 0
                profile_counter = 0

            epoch_ended = self.minibatch_idx == self.total_minibatch
            if (self.cur_minibatch % self.checkpoint_freq) == 0:
                # checkpoint every K minibatch
                self.update_checkpoint(model, optimizer, epoch_ended)
                # move back to device because exporting a model will move it
                # back to cpu and put it in eval mode
                model.to(device)
                model.train()

            start_stamp = time.time()

            if self.test_mode and self.minibatch_idx > self.total_minibatch:
                # early stopping for test mode
                break

        # reset index within an epoch
        self.minibatch_idx = 0


    def update_checkpoint(self, model, optimizer, epoch_ended):
        # move back model to cpu
        # and put in eval mode
        model.cpu()
        model.eval()

        # checkpoint every K minibatch
        checkpoint = {
            'epoch_idx': self.epoch_idx,
            'cur_minibatch': self.cur_minibatch,
            'minibatch_idx': self.minibatch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history,
            'accumulated_loss_counter': self.accumulated_loss_counter,
            'accumulated_loss': self.accumulated_loss,
            'epoch_ended': epoch_ended,
            'history_indices': self.history_indices,
        }

        checkpoint_file = os.path.join(
            self.log_dir,
            'checkpoint_{:09d}_{:09d}.dill'.format(self.cur_minibatch, self.epoch_idx)
        )
        with open(checkpoint_file, 'wb') as fid:
            dill.dump(checkpoint, fid)

        self.logger.info(f'save checkpoint to {checkpoint_file}')

        checkpoint_files = glob(self.log_dir + "/checkpoint_*.dill")
        checkpoint_files.sort()

        no_checkpoint_files = len(checkpoint_files)

        if self.max_checkpoint > 0 and no_checkpoint_files > self.max_checkpoint:
            for idx in range(0, no_checkpoint_files - self.max_checkpoint):
                os.remove(checkpoint_files[idx])

        # ----- handle onnx checkpoints --------------------------
        onnx_file = os.path.join(
            self.log_dir,
            'model_{:09d}_{:09d}.onnx'.format(self.cur_minibatch, self.epoch_idx)
        )
        self.export_to_onnx(model, self.sample_input, onnx_file)
        onnx_files = glob(self.log_dir + "/model_*.onnx")
        onnx_files.sort()
        nb_onnx_file = len(onnx_files)
        if self.max_checkpoint > 0 and nb_onnx_file > self.max_checkpoint:
            for idx in range(0, nb_onnx_file - self.max_checkpoint):
                os.remove(onnx_files[idx])


    def print_and_update(
        self,
        total_time,
        data_prep_time,
        data_to_gpu_time,
        forward_time,
        backward_time,
        profile_counter,
    ):
        avg_loss = self.accumulated_loss / self.accumulated_loss_counter
        self.loss_curve.write('{},{}\n'.format(self.cur_minibatch, avg_loss))

        # find number of digits of total train minibatch
        nb_digit = len(str(self.total_train_minibatch))
        formatter = '{:0' + str(nb_digit) + 'd}'
        cur_minibatch = formatter.format(self.cur_minibatch)

        nb_digit = len(str(self.epoch_idx + 1))
        formatter = '{:0' + str(nb_digit) + 'd}'
        epoch = formatter.format(self.epoch_idx + 1)

        print_content = [
            'minibatch: {} / {}| '.format(cur_minibatch, self.total_train_minibatch),
            'epoch: {}| '.format(epoch),
            'loss: {}| '.format(avg_loss),
            'avg latency per minibatch (in seconds): all={:.3f}, '.format(total_time / profile_counter),
            'data prep={:.3f}, '.format(data_prep_time / profile_counter),
        ]

        if self.move_data_to_device:
            print_content.append(
                'data to device={:.3f}, '.format(data_to_gpu_time / profile_counter),
            )
        print_content.append(
            'forward={:.3f}, '.format(forward_time / profile_counter),
        )
        print_content.append(
            'backward={:.3f}, '.format(backward_time / profile_counter),
        )

        estimated_time = self.get_estimated_time()
        print_content.append(
            'time taken: {:02d}:{:02d}:{:02d}| '.format(
                estimated_time['hour_taken'],
                estimated_time['minute_taken'],
                estimated_time['second_taken'],
            )
        )
        print_content.append(
            'time left: {:02d}:{:02d}:{:02d}| '.format(
                estimated_time['hour_left'],
                estimated_time['minute_left'],
                estimated_time['second_left'],
            )
        )

        self.logger.info(''.join(print_content))

        self.accumulated_loss_counter = 0
        self.accumulated_loss = 0


    def update_metrics(self, train_metrics: dict, val_metrics: dict, test_metrics: dict):
        # update indices
        self.history_indices.append(self.epoch_idx)

        # add new values to the history list
        prefixes = ['train_', 'val_', 'test_']
        values = [train_metrics, val_metrics, test_metrics]
        for prefix, value in zip(prefixes, values):
            for metric_name in value.keys():
                if prefix + metric_name in self.history:
                    self.history[prefix + metric_name].append(value[metric_name])
                else:
                    self.history[prefix + metric_name] = [value[metric_name]]


    def print_metrics(self, metrics, prefix):
        names = list(metrics.keys())
        names.sort()
        for name in names:
            value = self.get_metric_value(metrics[name])
            self.logger.info('{} {}: {:.6f}'.format(prefix, name, value))


    def get_estimated_time(self):
        minibatch_left = self.total_train_minibatch - self.cur_minibatch
        minibatch_taken = self.cur_minibatch - self.start_minibatch_idx
        time_taken = time.time() - self.start_time
        time_left = int(time_taken * minibatch_left / minibatch_taken)

        hour_left = int(time_left / 3600)
        minute_left = int((time_left - hour_left * 3600) / 60)
        second_left = int((time_left - hour_left * 3600 - minute_left * 60))

        hour_taken = int(time_taken / 3600)
        minute_taken = int((time_taken - hour_taken * 3600) / 60)
        second_taken = int((time_taken - hour_taken * 3600 - minute_taken * 60))
        return {
            'hour_taken': hour_taken,
            'minute_taken': minute_taken,
            'second_taken': second_taken,
            'hour_left': hour_left,
            'minute_left': minute_left,
            'second_left': second_left,
        }


    def update_tensorboard(self, tensorboard_logger, logger_prefix):
        names = list(self.history.keys())
        names.sort()
        if tensorboard_logger is not None:
            for name in names:
                if len(self.history[name]) > 0:
                    value = self.history[name][-1]
                    value = self.get_metric_value(value)
                    tensorboard_logger.add_scalar(
                        tag='{}/{}'.format(logger_prefix, name),
                        scalar_value=value,
                        global_step=self.epoch_idx + 1
                    )
                    tensorboard_logger.flush()

    def load_from_checkpoint(self, model, optimizer, device):
        ckp_files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith('checkpoint_')]
        ckp_files.sort()

        if self.checkpoint_idx == -1:
            # load from latest checkpoint
            if len(ckp_files) > 0:
                with open(ckp_files[-1], 'rb') as fid:
                    checkpoint = dill.load(fid)
            else:
                checkpoint = None
        elif self.checkpoint_idx is None:
            # train from scratch
            checkpoint = None
        elif self.checkpoint_idx >= 0:
            if len(ckp_files) == 0:
                self.logger.warning(f'no checkpoint exists; training will be done from scratch')
                checkpoint = None
            else:
                if self.checkpoint_idx < len(ckp_files):
                    ckp_file = ckp_files[self.checkpoint_idx]
                    with open(ckp_file, 'rb') as fid:
                        checkpoint = dill.load(fid)
                    self.logger.info(f'checkpoint file {ckp_file} loaded')
                else:
                    msg = (
                        f'invalid checkpoint index: {self.checkpoint_idx}; ',
                        f'there are only {len(ckp_files)} checkpoints'
                    )
                    raise RuntimeError(''.join(msg))
        else:
            raise RuntimeError(f'unknown checkpoint index: {self.checkpoint_idx}')

        if checkpoint is None:
            self.epoch_idx = 0
            self.history = {}
            self.history_indices = []
            self.loss_curve = open(os.path.join(self.output_dir, 'loss_curve.txt'), 'w')
            self.accumulated_loss_counter = 0
            self.accumulated_loss = 0
            self.cur_minibatch = 0
            self.minibatch_idx = 0
            progress_file_handle = open(os.path.join(self.output_dir, 'progress.txt'), 'w')
        else:
            # set the epoch index and previous metric values
            progress_file_handle = open(os.path.join(self.output_dir, 'progress.txt'), 'a')
            self.loss_curve = open(os.path.join(self.output_dir, 'loss_curve.txt'), 'a')
            self.cur_minibatch = checkpoint['cur_minibatch']
            self.minibatch_idx = checkpoint['minibatch_idx']
            self.epoch_idx = checkpoint['epoch_idx']
            self.history = checkpoint['history']
            self.accumulated_loss_counter = checkpoint['accumulated_loss_counter']
            self.accumulated_loss = checkpoint['accumulated_loss']
            self.history_indices = checkpoint['history_indices']

            if checkpoint['epoch_ended']:
                self.epoch_idx += 1
                self.minibatch_idx = 0

            # load model state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            # load optimizer state dict
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        if not isinstance(self.logger, Logger):
            self.logger = Logger(self.logger)

        self.logger.add_file_handle(progress_file_handle)
