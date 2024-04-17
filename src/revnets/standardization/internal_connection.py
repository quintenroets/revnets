from dataclasses import dataclass

import torch

from revnets import models

from .utils import Parameters, extract_parameters, extract_weights


@dataclass
class InternalConnection(models.InternalConnection):
    @property
    def input_weights(self) -> torch.Tensor:
        return extract_weights(self.input)

    @property
    def input_parameters(self) -> Parameters:
        return extract_parameters(self.input)

    @property
    def output_weights(self) -> torch.Tensor:
        return extract_weights(self.output)

    @property
    def output_parameters(self) -> Parameters:
        return extract_parameters(self.output)
