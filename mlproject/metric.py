"""
metric.py: metrics implementation
---------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import numpy as np
from loguru import logger


_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
_MSELoss = torch.nn.MSELoss()


class Metric(ABC):
    """
    Base metric class that provides an interface for all metric implementation
    """

    def __init__(self, name, **kwargs):
        self.metric_name = name
        pass

    @abstractmethod
    def update(self, predictions, labels):
        logger.warning("update() not implemented")
        pass

    @abstractmethod
    def value(self):
        logger.warning("value() not implemented")
        pass

    @abstractmethod
    def reset(self):
        logger.warning("reset() not implemented")
        pass

    def name(self):
        return self.metric_name

    def load(self) -> None:
        raise NotImplementedError(
            "load() not implemented, need to subclass & implement this method"
        )

    def dump(self) -> dict:
        raise NotImplementedError(
            "dump() not implemented, need to subclass & implement this method"
        )

    def merge(self, *others):
        raise NotImplementedError(
            "merge() not implemented, need to subclass & implement this method"
        )


class MSE(Metric):
    """
    Mean Squared Error metric
    """

    def __init__(self, name="mse", batch_axis=0):
        super(MSE, self).__init__(name=name)
        self._total_value = 0
        self._n_sample = 0
        self._batch_axis = batch_axis

    def dump(self) -> dict:
        return {"n_sample": self._n_sample, "total_value": self._total_value}

    def load(self, state: dict) -> None:
        self._total_value = state["total_value"]
        self._n_sample = state["n_sample"]

    def merge(self, *others):
        for other in others:
            self._total_value += other._total_value
            self._n_sample += other._n_sample

    def update(self, predictions, labels):
        """
        Update the value of mean squared error
        based on the current mini-batch's predictions and labels
        """
        n_sample = labels.shape[self._batch_axis]
        self._total_value += _MSELoss(predictions, labels).item() * n_sample
        self._n_sample += n_sample

    def value(self):
        """
        Return the MSE value
        """
        if self._n_sample > 0:
            return self._total_value / self._n_sample
        else:
            return 0.0

    def reset(self):
        self._total_value = 0
        self._n_sample = 0


class MAE(Metric):
    """
    Mean Absolute Error metric
    """

    def __init__(self, name="mae", batch_axis=0):
        super(MAE, self).__init__(name=name)
        self._total_value = 0
        self._n_sample = 0
        self._batch_axis = batch_axis

    def dump(self) -> dict:
        return {"n_sample": self._n_sample, "total_value": self._total_value}

    def load(self, state: dict) -> None:
        self._total_value = state["total_value"]
        self._n_sample = state["n_sample"]

    def merge(self, *others):
        for other in others:
            self._total_value += other._total_value
            self._n_sample += other._n_sample

    def update(self, predictions, labels):
        """
        Update the value of mean absolute error
        based on the current mini-batch's predictions and labels
        """
        n_sample = labels.shape[self._batch_axis]
        self._total_value += (
            torch.abs(predictions.flatten() - labels.flatten()).sum().item()
        )
        self._n_sample += n_sample

    def value(self):
        """
        Return the MAE value
        """
        if self._n_sample > 0:
            return self._total_value / self._n_sample
        else:
            return 0.0

    def reset(self):
        self._total_value = 0
        self._n_sample = 0


class Accuracy(Metric):
    """
    Classification Accuracy
    """

    def __init__(self, name="accuracy", class_index=None, confidence_threshold=None):
        super(Accuracy, self).__init__(name=name)

        if class_index is not None:
            if confidence_threshold is None:
                msg = (
                    "when class_index is specified, ",
                    "the confidence_threshold must also be specified",
                )
                raise RuntimeError("".join(msg))
            if not (0 <= confidence_threshold <= 1):
                raise RuntimeError("confidence_threshold must be in [0, 1]")

            msg = (
                "when class_index is specified, ",
                "the predictions given to update() will be softmax-normalized",
            )
            logger.warning("".join(msg))
            self.update_function = self.update_binary
        else:
            self.update_function = self.update_multiclass

        self._n_correct = 0
        self._n_sample = 0
        self._class_index = class_index
        self._confidence_threshold = confidence_threshold

    def dump(self) -> dict:
        return {
            "n_sample": self._n_sample,
            "n_correct": self._n_correct,
            "class_index": self._class_index,
            "confidence_threshold": self._confidence_threshold,
        }

    def load(self, state: dict) -> None:
        self._n_sample = state["n_sample"]
        self._n_correct = state["n_correct"]
        self._class_index = state["class_index"]
        self._confidence_threshold = state["confidence_threshold"]

    def merge(self, *others):
        for other in others:
            self._n_correct += other._n_correct
            self._n_sample += other._n_sample

    def update_binary(self, predictions, labels):
        """
        update function for binary accuracy computation
        """
        predictions = torch.softmax(predictions, -1)
        predictions = (
            (predictions[:, self._class_index] >= self._confidence_threshold)
            .flatten()
            .long()
        )
        # label is multi-class, need to convert to binary form
        labels = (labels == self._class_index).flatten().long()
        self._n_correct += (predictions == labels).sum().item()
        self._n_sample += predictions.size(0)

    def update_multiclass(self, predictions, labels):
        self._n_correct += (predictions.argmax(dim=-1) == labels).sum().item()
        self._n_sample += predictions.size(0)

    def update(self, predictions, labels):
        self.update_function(predictions, labels)

    def value(self):
        if self._n_sample > 0:
            return self._n_correct / self._n_sample
        else:
            return 0.0

    def reset(self):
        self._n_correct = 0
        self._n_sample = 0


class Precision(Metric):
    """
    Precision
    """

    def __init__(self, name="precision", class_index=None, confidence_threshold=None):
        super(Precision, self).__init__(name=name)

        if class_index is not None:
            if confidence_threshold is None:
                msg = (
                    "when class_index is specified, ",
                    "the confidence_threshold must also be specified",
                )
                raise RuntimeError("".join(msg))
            if not (0 <= confidence_threshold <= 1):
                raise RuntimeError("confidence_threshold must be in [0, 1]")

            msg = (
                "when class_index is specified, ",
                "the predictions given to update() will be softmax-normalized",
            )
            logger.warning("".join(msg))
            self.update_function = self.update_binary
            self.value_function = self.value_binary
        else:
            self.update_function = self.update_multiclass
            self.value_function = self.value_multiclass

        self._stat = {}
        self._n_class = 0
        self._n_sample = 0
        self._class_index = class_index
        self._confidence_threshold = confidence_threshold

    def dump(self) -> dict:
        data = {
            "n_sample": self._n_sample,
            "n_class": self._n_class,
            "class_index": self._class_index,
            "confidence_threshold": self._confidence_threshold,
        }

        for i in range(self._n_class):
            for key in self._stat[i].keys():
                data[f"true_pos_{i}"] = self._stat[i]["true_pos"]
                data[f"false_pos_{i}"] = self._stat[i]["false_pos"]
                data[f"false_neg_{i}"] = self._stat[i]["false_neg"]
        return data

    def load(self, state: dict) -> None:
        self._n_sample = state["n_sample"]
        self._n_class = state["n_class"]
        self._class_index = state["class_index"]
        self._confidence_threshold = state["confidence_threshold"]
        self._stat = {}
        for i in range(self._n_class):
            self._stat[i] = {
                "true_pos": state[f"true_pos_{i}"],
                "false_pos": state[f"false_pos_{i}"],
                "false_neg": state[f"false_neg_{i}"],
            }

    def merge(self, *others):
        for other in others:
            self._n_sample += other._n_sample
            for i in range(self._n_class):
                self._stat[i]["true_pos"] += other._stat[i]["true_pos"]
                self._stat[i]["false_pos"] += other._stat[i]["false_pos"]
                self._stat[i]["false_neg"] += other._stat[i]["false_neg"]

    def update(self, predictions, labels):
        self.update_function(predictions, labels)

    def update_multiclass(self, predictions, labels):
        if self._n_class == 0:
            # initialize for the 1st time calling update()
            self._n_class = predictions.size(1)
            self._stat = {
                i: {"true_pos": 0, "false_pos": 0, "false_neg": 0}
                for i in range(self._n_class)
            }

        self._n_sample += predictions.size(0)
        predictions = predictions.argmax(dim=-1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for i in range(self._n_class):
            pred_pos_index = np.where(predictions == i)[0]
            pred_neg_index = np.where(predictions != i)[0]
            label_pos_index = np.where(labels == i)[0]
            true_pos = np.intersect1d(pred_pos_index, label_pos_index).size
            false_neg = np.intersect1d(pred_neg_index, label_pos_index).size
            self._stat[i]["true_pos"] += true_pos
            self._stat[i]["false_pos"] += pred_pos_index.size - true_pos
            self._stat[i]["false_neg"] += false_neg

    def update_binary(self, predictions, labels):
        if self._n_class == 0:
            # initialize for the 1st time calling update()
            self._n_class = predictions.size(1)
            if self._class_index >= self._n_class:
                raise RuntimeError(
                    "given class_index exceeds the number of class in predictions"
                )
            self._stat = {"true_pos": 0, "false_pos": 0, "false_neg": 0}

        self._n_sample += predictions.size(0)
        # handle predictions for binary case
        predictions = torch.softmax(predictions, -1)
        predictions = (
            (predictions[:, self._class_index] >= self._confidence_threshold)
            .flatten()
            .long()
        )
        # handle labels for binary case
        labels = (labels == self._class_index).flatten().long()

        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        pred_pos_index = np.where(predictions == 1)[0]
        pred_neg_index = np.where(predictions != 1)[0]
        label_pos_index = np.where(labels == 1)[0]

        true_pos = np.intersect1d(pred_pos_index, label_pos_index).size
        false_neg = np.intersect1d(pred_neg_index, label_pos_index).size
        self._stat["true_pos"] += true_pos
        self._stat["false_pos"] += pred_pos_index.size - true_pos
        self._stat["false_neg"] += false_neg

    def value_multiclass(self):
        if self._n_sample > 0:
            values = []
            for i in range(self._n_class):
                denominator = self._stat[i]["true_pos"] + self._stat[i]["false_pos"]
                if denominator == 0:
                    logger.warning(f"Zero in the denominator of precision in class {i}")
                    values.append(0)
                else:
                    nominator = self._stat[i]["true_pos"]
                    values.append(nominator / denominator)
            return np.mean(values)
        else:
            return 0.0

    def value_binary(self):
        if self._n_sample > 0:
            denominator = self._stat["true_pos"] + self._stat["false_pos"]
            if denominator == 0:
                logger.warning(f"Zero in the denominator of precision")
                return 0.0
            else:
                nominator = self._stat["true_pos"]
                return nominator / denominator
        else:
            return 0.0

    def value(self):
        return self.value_function()

    def reset(self):
        self._stat = {}
        self._n_class = 0
        self._n_sample = 0


class Recall(Precision):
    """
    Recall
    """

    def __init__(self, name="recall", class_index=None, confidence_threshold=None):
        super(Recall, self).__init__(
            name=name,
            class_index=class_index,
            confidence_threshold=confidence_threshold,
        )

    def value_multiclass(self):
        if self._n_sample > 0:
            values = []
            for i in range(self._n_class):
                denominator = self._stat[i]["true_pos"] + self._stat[i]["false_neg"]
                if denominator == 0:
                    logger.warning(f"Zero in the denominator of recall in class {i}")
                    values.append(0)
                else:
                    nominator = self._stat[i]["true_pos"]
                    values.append(nominator / denominator)
            return np.mean(values)
        else:
            return 0.0

    def value_binary(self):
        if self._n_sample > 0:
            denominator = self._stat["true_pos"] + self._stat["false_neg"]
            if denominator == 0:
                logger.warning(f"Zero in the denominator of recall")
                return 0.0
            else:
                nominator = self._stat["true_pos"]
                return nominator / denominator
        else:
            return 0.0


class F1(Precision):
    """
    F1 metric
    """

    def __init__(self, name="f1", class_index=None, confidence_threshold=None):
        super(F1, self).__init__(
            name=name,
            class_index=class_index,
            confidence_threshold=confidence_threshold,
        )

    def value_multiclass(self):
        if self._n_sample > 0:
            values = []
            for i in range(self._n_class):
                # compute recall
                recall_denominator = (
                    self._stat[i]["true_pos"] + self._stat[i]["false_neg"]
                )
                if recall_denominator == 0:
                    logger.warning(f"Zero in the denominator of recall in class {i}")
                    recall = 0
                else:
                    recall_nominator = self._stat[i]["true_pos"]
                    recall = recall_nominator / recall_denominator

                # compute precision
                precision_denominator = (
                    self._stat[i]["true_pos"] + self._stat[i]["false_pos"]
                )
                if precision_denominator == 0:
                    logger.warning(f"Zero in the denominator of precision in class {i}")
                    precision = 0
                else:
                    precision_nominator = self._stat[i]["true_pos"]
                    precision = precision_nominator / precision_denominator

                if abs(precision + recall) <= 1e-7:
                    values.append(0.0)
                else:
                    values.append(2 * precision * recall / (precision + recall))

            return np.mean(values)
        else:
            return 0.0

    def value_binary(self):
        if self._n_sample > 0:
            # compute recall
            recall_denominator = self._stat["true_pos"] + self._stat["false_neg"]
            if recall_denominator == 0:
                logger.warning("Zero in the denominator of recall")
                recall = 0
            else:
                recall_nominator = self._stat["true_pos"]
                recall = recall_nominator / recall_denominator

            # compute precision
            precision_denominator = self._stat["true_pos"] + self._stat["false_pos"]
            if precision_denominator == 0:
                logger.warning("Zero in the denominator of precision")
                precision = 0
            else:
                precision_nominator = self._stat["true_pos"]
                precision = precision_nominator / precision_denominator

            if abs(precision + recall) <= 1e-7:
                return 0.0
            else:
                return 2 * precision * recall / (precision + recall)

        else:
            return 0.0


class CrossEntropy(Metric):
    """
    Cross Entropy
    """

    def __init__(self, name="cross_entropy", batch_axis=0):
        super(CrossEntropy, self).__init__(name=name)
        self._total_value = 0
        self._n_sample = 0
        self._batch_axis = batch_axis

    def dump(self) -> dict:
        return {"n_sample": self._n_sample, "total_value": self._total_value}

    def load(self, state: dict) -> None:
        self._total_value = state["total_value"]
        self._n_sample = state["n_sample"]

    def merge(self, *others):
        for other in others:
            self._total_value += other._total_value
            self._n_sample += other._n_sample

    def update(self, predictions, labels):
        """
        Update the value of mean absolute error
        based on the current mini-batch's predictions and labels
        """
        if labels.size() == (1, 1):
            # note that (1, 1) label shape for CE should be (1,) only
            labels = labels.squeeze(0)

        n_sample = labels.shape[self._batch_axis]
        self._total_value += _CrossEntropyLoss(predictions, labels).item() * n_sample
        self._n_sample += n_sample

    def value(self):
        if self._n_sample > 0:
            return self._total_value / self._n_sample
        else:
            return 0.0

    def reset(self):
        self._total_value = 0
        self._n_sample = 0


class MetricFromLoss(Metric):
    """
    Metric class for the corresponding loss function
    """

    def __init__(self, name, loss_func, batch_axis=0):
        super(MetricFromLoss, self).__init__(name=name)
        self._loss_func = loss_func
        self._total_value = 0
        self._n_sample = 0
        self._batch_axis = batch_axis

        logger.warning(
            f"{self.__class__.__name__} initialized. Assuming batch axis is {self._batch_axis}."
            "Ensure your data is shaped accordingly."
        )

    def dump(self) -> dict:
        return {"n_sample": self._n_sample, "total_value": self._total_value}

    def load(self, state: dict) -> None:
        self._total_value = state["total_value"]
        self._n_sample = state["n_sample"]

    def merge(self, *others):
        for other in others:
            self._total_value += other._total_value
            self._n_sample += other._n_sample

    def update(self, predictions, labels):
        """
        Update the value of mean absolute error
        based on the current mini-batch's predictions and labels
        """
        n_sample = labels.shape[self._batch_axis]
        self._total_value += self._loss_func(predictions, labels).item() * n_sample
        self._n_sample += n_sample

    def value(self):
        if self._n_sample > 0:
            return self._total_value / self._n_sample
        else:
            return 0.0

    def reset(self):
        self._total_value = 0
        self._n_sample = 0


class NestedMetric(Metric):
    """
    Convert a list of metrics to a single metric interface
    """

    def __init__(self, name, metrics, weights=None):
        super(NestedMetric, self).__init__(name=name)
        self._metrics = metrics
        self._weights = weights

    def update(self, predictions, labels):
        """
        note here that predictions and labels can be in the nested form
        """
        self._recursive_update(predictions, labels, self._metrics)

    def _recursive_update(self, predictions, labels, metrics):
        if isinstance(predictions, (list, tuple)):
            for pred, lb, m in zip(predictions, labels, metrics):
                self._recursive_update(pred, lb, m)
        else:
            if metrics is not None:
                metrics.update(predictions, labels)

    def value(self):
        total_value = 0
        if self._weights is None:
            return self._recursive_value(self._metrics, None, total_value)
        else:
            return self._recursive_value(self._metrics, self._weights, total_value)

    def _recursive_value(self, metrics, weights, value):
        if isinstance(metrics, (list, tuple)):
            if weights is None:
                for m in metrics:
                    value = self._recursive_value(m, None, value)
            else:
                for m, w in zip(metrics, weights):
                    value = self._recursive_value(m, w, value)
            return value
        else:
            if metrics is None:
                return value
            if weights is None:
                return value + metrics.value()
            else:
                return value + weights * metrics.value()

    def _recursive_printable_value(self, metrics, value, indent):
        if isinstance(metrics, (list, tuple)):
            for m in metrics:
                value = self._recursive_printable_value(m, value, indent)
            return value
        else:
            if metrics is None:
                return value
            else:
                return value + metrics.get_printable_value(indent)

    def get_printable_value(self, indent=""):
        value = ""
        return self._recursive_printable_value(self.metrics, value, indent)

    def reset(self):
        self._recursive_reset(self._metrics)

    def _recursive_reset(self, metrics):
        if isinstance(metrics, (list, tuple)):
            for m in metrics:
                self._recursive_reset(m)
        else:
            if metrics is not None:
                metrics.reset()


def torch_tensor_to_numpy(x):
    """
    This function is used to convert
    nested list of torch tensors to a nested list of numpy tensor
    """
    # convert nested list of torch tensor to numpy
    if isinstance(x, (list, tuple)):
        x = [torch_tensor_to_numpy(item) for item in x]
    else:
        x = x.cpu().detach().numpy()

    return x


def concatenate_list(inputs):
    """
    This function is used to concatenate
    a list of nested lists of numpy array to a nested list of numpy arrays

    For example,
    inputs = [x_1, x_2, ..., x_N]
    each x_i is a nested list of numpy arrays
    for example,
    x_i = [np.ndarray, [np.ndarray, np.ndarray]]

    the outputs should be
    outputs = [np.ndarray, [np.ndarray, np.ndarray]]
    in which each np.ndarray is a concatenation of the corresponding element
    from the same position
    """

    def _create_nested_list(x, data):
        if isinstance(x, (list, tuple)):
            # there is a nested structure
            # create sub list for each item
            for _ in range(len(x)):
                data.append([])
            for idx, item in enumerate(x):
                _create_nested_list(item, data[idx])
        else:
            # do nothing to the input list
            next

    def _process_sample(x, data):
        if isinstance(x, (list, tuple)):
            for idx, item in enumerate(x):
                _process_sample(item, data[idx])
        else:
            data.append(x)

    # create an empty nested list based on the structure of the 1st sample
    outputs = []
    _create_nested_list(inputs[0], outputs)

    # then process each sample
    for sample in inputs:
        _process_sample(sample, outputs)

    def _concatenate_item(x):
        if isinstance(x, list) and isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        elif isinstance(x, list) and isinstance(x[0], list):
            return [_concatenate_item(item) for item in x]

    return _concatenate_item(outputs)


def _find_batch_size(predictions):
    if isinstance(predictions, (list, tuple)):
        return _find_batch_size(predictions[0])
    else:
        return predictions.size(0)
