import torch
import torch.optim as optim
from torch import nn

from ...context import context
from ...models import Activation
from ...utils import NamedClass


class Model(torch.nn.Module, NamedClass):
    def __init__(
        self, learning_rate: float | None = None, activation: Activation = None
    ) -> None:
        super().__init__()
        self.learning_rate = (
            learning_rate or context.config.target_network_training.learning_rate
        )
        self.activation_function = self.get_activation_function(activation)

    @classmethod
    def get_base_name(cls):
        return Model.__module__

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @classmethod
    def get_activation_function(cls, activation: Activation = None):
        if activation is None:
            activation = context.config.target_network_training.activation
        match activation:
            case Activation.leaky_relu:
                activation_function = nn.functional.leaky_relu
            case Activation.relu:
                activation_function = nn.functional.relu
            case Activation.tanh:
                activation_function = nn.functional.tanh
        return activation_function  # noqa
