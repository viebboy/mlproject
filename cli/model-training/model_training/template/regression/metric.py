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

from mlproject.metric import MetricFromLoss
from mlproject.loss import MSE as MSE_LOSS
from mlproject.loss import MAE as MAE_LOSS


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
        MetricFromLoss(name="mse", loss_func=MSE_LOSS),
        MetricFromLoss(name="mae", loss_func=MAE_LOSS),
    ]  # some metric objects here
    monitor_metric = "mse"  # name of the monitor metric here
    monitor_direction = (
        "lower"  # direction of the monitor metric here, can be higher or lower
    )

    return {
        "metrics": metrics,
        "monitor_metric": monitor_metric,
        "monitor_direction": monitor_direction,
    }
