from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from . import rnn


class ExtractRNNOutput(rnn.ExtractRNNOutput):
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        outputs, hidden = inputs
        return outputs


@dataclass
class NetworkFactory(rnn.NetworkFactory):
    def create_layers(self) -> Iterable[torch.nn.Module]:
        yield from (
            rnn.CreateRNNInput(),
            nn.RNN(self.input_size, self.input_size, batch_first=True, num_layers=3),
            ExtractRNNOutput(),
        )
        yield from super().create_layers()
