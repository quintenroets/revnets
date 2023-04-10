import sys
from dataclasses import dataclass

import cli

from ..utils import Path
from . import data_requirements, experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self):
        self.start_experiment()
        while self.get_failure_path() is not None:
            print(self.get_failure_path())
            self.start_experiment()

    @classmethod
    def start_experiment(cls):
        args = cls.get_args()
        cli.run("revnets", *args)

    def get_failure_path(self):
        glob_pattern = f"*{data_requirements.Experiment.failure_keyword}*"
        result_path = Path.results / self.name
        failure_paths = result_path.rglob(glob_pattern)
        return next(failure_paths, None)

    @classmethod
    @property
    def name(cls):  # noqa
        return data_requirements.Experiment.name

    @classmethod
    def get_args(cls):
        args = sys.argv[1:]

        experiment_option = "--experiment"
        experiment_value = "data_requirements"

        if experiment_option in args:
            experiment_option_index = args.index(experiment_option)
            args[experiment_option_index + 1] = experiment_value
        else:
            args += [experiment_option, experiment_value]
        return args
