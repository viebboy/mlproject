# mlproject.metric Module

The metric module provides a base interface for a metric class, as well as popular metrics.

A metric that can be used with `mlproject.trainer.Trainer` must expose the following methods:

- `update(predictions, labels)`: update the computation of metrics with the outputs from the model and the corresponding labels.
  The predictions and labels can come from minibatches of data, rather than coming from the whole dataset.
- `value()`: return the value of the metric
- `reset()`: reset the metric by flushing out all the recorded statistics
- `name()`: return the name of the metric

The above interface allows the trainer to compute a metric in a progressive manner, without having to pass the entire predictions and labels of a dataset.

This is especially useful when working with large datasets because limitation in memory prohibits us from keeping the whole list of predictions or labels.

`mlproject.metric` module provides implementation of popular metrics using the above interface and these implementations are memory-efficient. 

The implemented metrics include:

- `CrossEntropy`
- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- `MAE`
- `MSE`

When creating a metric object from the above metrics, one could also pass user-defined name for the metric to avoid name clashing in the Trainer class.

In order to implement a custom metric, one needs to subclass `mlproject.metric.Metric` and implement 4 methods mentioned above.

In addition, `mlproject.metric` also provides the following convenient interface:

## mlproject.metric.MetricFromLoss

This interface allows converting a loss function to a metric object that exposes required interfaces of `mlproject.metric.Metric`. 

This only works with those loss functions that allow averaging over minibatches such as mean-squared error or mean absolute eror.

Accuracy, precision, recall and f1 are, for example, cannot be computed by simply averaging the accuracy, precision, recall or f1 of different minibatches.

They requires aggregating rather than averaging
