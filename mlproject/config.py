"""
config.py: configuration related tools for mlproject
----------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-01
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache 2.0 License


"""

import itertools
import pprint


class ConfigValue(object):
    """
    Configuration abstraction used in project template
    """
    def __init__(self, *argv):
        self._value = []
        for v in argv:
            self._value.append(v)
        self._index = -1

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        self._index += 1
        if self._index < len(self._value):
            return self._value[self._index]
        raise StopIteration


def create_all_config(config_set):
    names = sorted(config_set.keys())
    all_values = []
    for name in names:
        all_values.append(config_set[name])
    configs = list(itertools.product(*all_values))
    outputs = []
    for conf in configs:
        # name is the name of the parameter
        # conf is a list of values for those parameters in names
        item = {name: conf[idx] for idx, name in enumerate(names)}
        if item not in outputs:
            outputs.append(item)
    return outputs


def print_all_config(configs):
    for idx, item in enumerate(configs):
        msg = f'config_index={idx}'
        print('-' * len(msg))
        print(msg)
        print('-' * len(msg))
        pprint.pprint(item)
