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
            self.start_experiment()

    @classmethod
    def start_experiment(cls):
        args = ("--experiment", "data_requirements", *sys.argv[1:])
        cli.run("revnets", *args, check=False)

    def get_failure_path(self):
        glob_pattern = f"*{data_requirements.Experiment.failure_keyword}*"
        result_path = Path.results / self.name
        failure_paths = result_path.rglob(glob_pattern)
        return next(failure_paths, None)

    @classmethod
    @property
    def name(cls):  # noqa
        return data_requirements.Experiment.name
