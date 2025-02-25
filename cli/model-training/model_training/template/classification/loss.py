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

import torch.nn as nn


# this method should return the loss function
# if you use standard loss function by pytorch, then simply return them here
# also good to take a look at mlproject.loss module
def get_loss_function(**arguments: dict) -> callable:
    # here we return cross entropy loss
    return nn.CrossEntropyLoss()
