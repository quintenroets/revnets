from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from . import mininet


class CreateRNNInput(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.squeeze()


class ExtractRNNOutput(nn.Module):
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        number_of_dimensions_without_batch = 2
        outputs, hidden = inputs
        if outputs.dim() == number_of_dimensions_without_batch:
            outputs = outputs.unsqueeze(0)
        return outputs[:, -1, :]


@dataclass
class NetworkFactory(mininet.NetworkFactory):
    input_size: int = 28
    sequence_length: int = 28
    input_shape: tuple[int, ...] = (sequence_length, input_size)

    def create_layers(self) -> Iterable[torch.nn.Module]:
        return (
            CreateRNNInput(),
            nn.RNN(self.input_size, self.hidden_size, batch_first=True, num_layers=3),
            ExtractRNNOutput(),
            nn.Linear(self.hidden_size, self.output_size),
        )
