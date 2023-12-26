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

from __future__ import annotations
import os
import itertools
import pprint
from tabulate import tabulate
from git import Repo, InvalidGitRepositoryError


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

    def __len__(self):
        return len(self._value)


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
        msg = f"config_index={idx}"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
        pprint.pprint(item)


def cut_string(text, max_length=120):
    outputs = []
    current_line = []
    words = text.split(" ")
    for word in words:
        if (
            len(
                " ".join(
                    current_line
                    + [
                        word,
                    ]
                )
            )
            < max_length
        ):
            current_line.append(word)
        else:
            outputs.append(" ".join(current_line))
            current_line = []

    if len(current_line) > 0:
        outputs.append(" ".join(current_line))

    return "\n".join(outputs) + "\n"


def get_config_string(config: dict):
    config_string = ""
    config_string += f"CONFIGURATION INDEX: {config['config_index']}\n"
    config_string += "--------------------------------------\n"
    names = list(config.keys())
    table = [[name, cut_string(str(config[name]))] for name in names]
    table = tabulate(table, headers=["Config Name", "Config Value"])
    config_string += table
    return config_string


def get_repo_info(file_path: str):
    # Verify if the file path is valid
    if not os.path.isfile(file_path):
        return "The given path is not a valid file."

    try:
        # Initialize a repo object from the file path
        # This will find the root of the Git repository
        repo = Repo(file_path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return "Not a Git repository or the file is not in a Git repository."

    # Get the Git root directory
    repo_root = repo.git.rev_parse("--show-toplevel")

    # Get the current branch name
    branch = repo.active_branch.name

    # Get the remote URL (assuming 'origin' remote)
    remote_url = repo.remotes.origin.url

    # Get the latest commit hash for the specified file
    # Using a relative path from the repo root
    relative_file_path = os.path.relpath(file_path, repo_root)
    latest_commit = repo.git.log("-1", "--format=%H", relative_file_path)

    return {"git_url": remote_url, "git_branch": branch, "git_commit": latest_commit}
