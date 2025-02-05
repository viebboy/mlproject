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


# define get_metric() here
# should return a dict that contains "metrics", "monitor_metric", "monitor_direction"
def get_metric(**arguments: dict) -> dict:
    metrics = []  # some metric objects here
    monitor_metric = ""  # name of the monitor metric here
    monitor_direction = (
        "higher"  # direction of the monitor metric here, can be higher or lower
    )

    return {
        "metrics": metrics,
        "monitor_metric": monitor_metric,
        "monitor_direction": monitor_direction,
    }
