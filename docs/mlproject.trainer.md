# mlproject.trainer Module

This module provides the training orchestration. 

## mlproject.trainer.Trainer

This is the trainer class that orchestrates the training loop, evaluation, saving checkpoints, automatic checkpoint loading and so on.

The constructor of `Trainer` looks like this

```python
Trainer(
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
)
```

where:
- `n_epoch`: defines the total number of training epoch
- `output_dir`: the path to the directory where results will be saved
- `loss_function`: the loss function used to compute the cost of training from the model's predictions and the labels
- `metrics`: this is a list that contains the metric objects used to evaluate the performance
- `monitor_metric`: the name of the metric that should be used to monitor the performance
- `monitor_direction`: 'higher` means the higher the better and `lower` means lower the better.
   For example, if monitor metric is accuracy then the direction should be higher. For MSE, it is the other way around.
- `checkpoint_idx`: the index of the checkpoint to resume. Checkpoints are saved in `log_dir` if specified.
- `lr_scheduler`: the function that returns the learning rate value based on the current epoch index and the total number of epoch.
   Default to a cosine learning rate scheduler that reduces from 1e-3 to 1e-5 over the course of `n_epoch`
- `optimizer`: the name of the optimzer to be used. Supported `adam`, `sgd`, `adamW`.
- `weight_decay`: magnitude of weight decay regularization
- `log_dir`: path to the log directory that contains intermediate checkpoints
- `checkpoint_freq`: frequency that checkpoint is generated. The unit is minibatches.
  Default to 10. This means checkpoint is created every 10 minibatches.
    max_checkpoint: int=10,
    eval_freq: int=1,
    print_freq: int=10,
    use_progress_bar: bool=True,
    test_mode: bool=False,
    move_data_to_device: bool=True,
    retain_metric_objects: bool=True,
    sample_input=None,
    logger=guru_logger,
- `max_checkpoint`: the total number of checkpoints that will be retained. Checkpoints are pruned from the oldest and only the most recent ones are kept.
   A value of -1 means retaining all checkpoints
- `eval_freq`: the frequency that model is evaluated using the given metrics. Unit is epoch.
   Default to 1, meaning model is evaluated at the end of each epoch.
- `print_freq`: the frequency to print the training loss value. Unit is minibatches.
   Default to 10, meaning printing is done for every 10 minibatches
- `use_progress_bar`: whether to show progress bar when during evaluation.
- `test_mode`: if test mode is enabled, each epoch is only run for maximum of 100 minibatches
- `move_data_to_device`: if true, data is also moved to the user-specified device
- `retain_metric_objects`: if true, the history will contain copy of metric objects at each evaluation. Otherwise, only the metric value returned by `.value()` is saved.
   The metric objects could contain more information than a single value.
- `sample_input`: if given, this will be used as the sample input in ONNX model generation. Otherwise, sample input is inferred from the training data.
- `logger`: logger object

After creating a trainer object, we could use it to optimize a model by calling the fit function as follows:

```python
trainer.fit(
    model,
    train_data,
    val_data=None,
    test_data=None,
    device=torch.device('cpu'),
    tensorboard_logger=None,
    logger_prefix='',
)

where:

- `model` is the pytorch model to be optimized
- `train_data` is a dictionary that should contain at least `dataloader` as a key. 
   `train_data["dataloader"]` should return the data loader objects.
- `val_data`: same as train data. Default to None. If not None, it will be used for model selection (validation purposes)
- `test_data`: same as train data. Default to None. If not None, performance will also be measured for this dataset.
- `device`: the torch device that is used for computation. Default to `torch.device('cpu')`
- `tensorboard_logger`: the tensorboard logger object. Default to None.
- `logger_prefix`: the prefix or subsection in the tensorboard logger.


## Custom Trainer

When implementing a custom trainer, one should subclass `mlproject.trainer.Trainer` class and potentially overwrite the following methods:

- `update_loop()`: the method provides the update logics of one epoch. 
  One should only change how the loss is computed inside this method. 
  In the generic template, the original code of this method has been copied for references.
- `eval()`: the evaluation logics. One should only change the metric updating procedures inside this function.
  In the generic template, the original code of this method has been copied for references.
- `export_to_onnx()`: this contains the model export logics. At the moment, this method only works with single-input model.
  One should overwrite this method to provide proper arguments to `torch.onnx.export()` if one wants to fix the batch dimension or
  support multi-input model etc.
  For multi-input model, the easiest solution is to wrap all inputs into a list like `model([input1, input2])` instead of implementing a model
  that makes prediction like `model(input1, input2)`
  the former is effectively single-input model and the latter is effectively multi-input model. 
