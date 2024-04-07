from functools import cached_property

import torch
from package_utils.context import Context as Context_
from torch import nn

from ..models import Activation, Config, HyperParameters, Options, Path


class Context(Context_[Options, Config, None]):
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

    @property
    def activation_layer(self) -> nn.Module:
        activation = context.config.target_network_training.activation
        match activation:
            case Activation.leaky_relu:
                activation_layer = nn.LeakyReLU()
            case Activation.relu:
                activation_layer = nn.ReLU()
            case Activation.tanh:
                activation_layer = nn.Tanh()
        return activation_layer

    @cached_property
    def device(self) -> torch.device:
        name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)


context = Context(Options, Config, None)
