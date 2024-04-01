from functools import cached_property

from package_utils.context import Context as Context_

from ..models import Config, HyperParameters, Options, Path, Secrets


class Context(Context_[Options, Config, Secrets]):
    @property
    def training(self) -> HyperParameters:
        return (
            self.config.reconstruction_training_debug
            if self.config.debug
            else self.config.reconstruction_training
        )

    @property
    def number_of_epochs(self) -> int:
        return self.training.epochs

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @cached_property
    def is_running_in_notebook(self):
        try:
            get_ipython()
            is_in_notebook = True
        except NameError:
            is_in_notebook = False
        return is_in_notebook

    @property
    def output_path(self) -> Path:
        return self.config.experiment.path

    @property
    def results_path(self) -> Path:
        path = self.output_path / "results.yaml"
        return path.with_nonexistent_name()

    @property
    def log_path(self) -> Path:
        return self.output_path / "logs"

    @property
    def log_path_str(self) -> str:
        return str(self.log_path)


context = Context(Options, Config, Secrets)
