"""
loss.py: loss function implementations
--------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

import torch
import torch.nn.functional as F


_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
_MSELoss = torch.nn.MSELoss()


def get_MSE(weights=None):
    """
    Get MSE loss function with the possibility to specify the weight for each component
    MSE loss is supported for both single and multi-output model
    With multi-output model, individual losses are aggregated
    """
    def _MSE(predictions, labels, loss=0, W=weights):
        if isinstance(predictions, (list, tuple)):
            # multi output case
            if W is None:
                for pred, lb in zip(predictions, labels):
                    loss = _MSE(pred, lb, loss, None)
            else:
                for pred, lb, w in zip(predictions, labels, W):
                    loss = _MSE(pred, lb, loss, w)
        else:
            if W is None:
                loss = loss + _MSELoss(predictions, labels)
            else:
                loss = loss + W * _MSELoss(predictions, labels)

        return loss

    return _MSE


# standard MSE loss without weights
MSE = get_MSE()


def get_MAE(weights=None):
    """
    Get MAE loss function with the possibility to specify the weight for each component
    MAE loss is supported for both single and multi-output model
    With multi-output model, individual losses are aggregated
    """

    def _mae(predictions, labels):
        return torch.abs(predictions.flatten() - labels.flatten()).mean()

    def _MAE(predictions, labels, loss=0, W=weights):
        if isinstance(predictions, (list, tuple)):
            # multi output case
            if W is None:
                for pred, lb in zip(predictions, labels):
                    loss = _MAE(pred, lb, loss, None)
            else:
                for pred, lb, w in zip(predictions, labels, W):
                    loss = _MAE(pred, lb, loss, w)
        else:
            if W is None:
                loss = loss + _mae(predictions, labels)
            else:
                loss = loss + W * _mae(predictions, labels)

        return loss

    return _MAE


# standard MAE loss without weights
MAE = get_MAE()


def get_CrossEntropy(weights=None):
    """
    Get cross entropy loss function with the possibility to specify the weight for each component
    cross entropy loss is supported for both single and multi-output model
    With multi-output model, individual losses are aggregated
    """
    def _CrossEntropy(predictions, labels, loss=0, W=weights):
        if isinstance(predictions, (list, tuple)):
            # multi output case
            if W is None:
                for pred, lb in zip(predictions, labels):
                    loss = _CrossEntropy(pred, lb, loss, None)
            else:
                for pred, lb, w in zip(predictions, labels, W):
                    loss = _CrossEntropy(pred, lb, loss, w)
        else:
            if W is None:
                loss = loss + _CrossEntropyLoss(predictions, labels.flatten())
            else:
                loss = loss + W * _CrossEntropyLoss(predictions, labels)

        return loss

    return _CrossEntropy


# standard CrossEntropy loss without weights
CrossEntropy = get_CrossEntropy()

def CosineDissimilarity(predictions, labels):
    # predictions: N x D
    # labels: N x D
    batch_size = predictions.size(0)
    # normalize
    predictions = predictions / torch.norm(predictions, p='fro', dim=-1, keepdim=True)
    labels = labels / torch.norm(labels, p='fro', dim=-1, keepdim=True)
    return 1 - torch.trace(torch.matmul(predictions, labels.T)) / batch_size


def get_HintonCrossEntropy(**kwargs):
    def kd_loss(predictions, labels, ground_truth=None):
        loss = torch.nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(predictions / kwargs['temperature'], dim=1),
                F.softmax(labels / kwargs['temperature'], dim=1)
        )
        if ground_truth is not None:
            loss = (
                loss * kwargs['alpha'] * kwargs['temperature'] * kwargs['temperature'] +
                F.cross_entropy(predictions, ground_truth) * (1 - kwargs['alpha'])
            )
        return loss


def compose_losses_to_callable(nested_loss, weights=None):
    """
    Convert a nested list of loss functions to a single loss function callable
    Users can also specify the same nested structure for the weight multiplier for each loss
    This utility helps us to construct a loss_func for nested-output model
    """
    def loss_func(
        predictions,
        labels,
        loss_functions=nested_loss,
        total_loss=0,
        W=weights
    ):
        if isinstance(predictions, (list, tuple)):
            # multi output case
            if W is None:
                for pred, lb, func in zip(predictions, labels, loss_functions):
                    total_loss = loss_func(pred, lb, func, total_loss, None)
            else:
                for pred, lb, func, w in zip(predictions, labels, loss_functions, W):
                    total_loss = loss_func(pred, lb, func, total_loss, w)
        else:
            if W is None:
                total_loss = total_loss + loss_functions(predictions, labels)
            else:
                total_loss = total_loss + W * loss_functions(predictions, labels)

        return total_loss

    return loss_func
