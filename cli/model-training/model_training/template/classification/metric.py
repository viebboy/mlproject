"""
metric.py: metric implementation
--------------------------------


* Copyright: Dat Tran (viebboy@gmail.com)
* Authors: Dat Tran
* Date: #TODO
* Version: 0.0.1

This is part of the #TODO project

License
-------
Proprietary License

"""

from mlproject.metric import Accuracy, Precision, Recall, F1


# define get_metric() here
# should return a dict that contains "metrics", "monitor_metric", "monitor_direction"
def get_metric(**arguments: dict) -> dict:
    """
    this function should return a dict that contains "metrics", "monitor_metric", "monitor_direction"
    metrics: a list of metric objects that adopts the interface in mlproject.metric.Metric
    monitor_metric: name of the monitor metric here
    monitor_direction: direction of the monitor metric here, can be higher or lower
    """
    metrics = [
        Accuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        F1(name="f1"),
    ]
    monitor_metric = "f1"
    monitor_direction = "higher"

    return {
        "metrics": metrics,
        "monitor_metric": monitor_metric,
        "monitor_direction": monitor_direction,
    }
