"""
loss.py: loss implementation
----------------------------


* Copyright: Dat Tran (viebboy@gmail.com)
* Authors: Dat Tran
* Date: #TODO
* Version: 0.0.1

This is part of the #TODO project

License
-------
Proprietary License

"""

from mlproject.loss import MSE, MAE


# this method should return the loss function
# if you use standard loss function by pytorch, then simply return them here
# also good to take a look at mlproject.loss module
def get_loss_function(**arguments: dict) -> callable:
    """
    this should return a callable that takes in predictions and labels and returns a loss value
    """

    if arguments["loss_type"] == "mse":
        return MSE()
    elif arguments["loss_type"] == "mae":
        return MAE()
    else:
        raise ValueError(f"Loss type {arguments['loss_type']} not supported")
