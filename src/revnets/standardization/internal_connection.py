from dataclasses import dataclass

from revnets import models

from .parameters import Parameters, extract_parameters


@dataclass
class InternalConnection(models.InternalConnection):
    @property
    def input_parameters(self) -> Parameters:
        return extract_parameters(self.input)

    @property
    def output_parameters(self) -> Parameters:
        return extract_parameters(self.output)
