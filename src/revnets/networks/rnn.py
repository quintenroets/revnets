from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from . import mininet


class CreateRNNInput(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
        a = inputs.unsqueeze(2)
        pprint(a.shape)
        return inputs.unsqueeze(2)


class ExtractRNNOutput(nn.Module):
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        outputs, hidden = inputs
        last_outputs = outputs[:, -1, :]
        return last_outputs


@dataclass
class NetworkFactory(mininet.NetworkFactory):
    input_size: int = 13
    input_shape: tuple[int, ...] = (17, input_size)

    def create_layers(self) -> Iterable[torch.nn.Module]:
        return (
            CreateRNNInput(),
            nn.RNN(self.input_size, self.hidden_size, batch_first=True, num_layers=3),
            ExtractRNNOutput(),
            nn.Linear(self.hidden_size, self.output_size),
        )
