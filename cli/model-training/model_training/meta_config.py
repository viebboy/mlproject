# abstraction for meta config when running model search
import optuna
from abc import abstractmethod


class MetaConfig:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_config_file(self, trial: optuna.trial.Trial) -> str:
        # generate a config file by sampling the space with trial
        # return config path and output folder path
        pass

    @abstractmethod
    @property
    def nb_trials(self) -> int:
        # return the number of trials
        pass

    @abstractmethod
    @property
    def nb_parallel_trials(self) -> int:
        pass

    @abstractmethod
    def get_score(self, output_dir: str) -> float:
        # override to load output file and get corresponding score for an experiment
        pass

    @abstractmethod
    def db_path(self) -> str:
        # return the path to the sqlite database used to keep track of the trials
        pass
