from abc import ABC
from dataclasses import dataclass

import torch


@dataclass
class Weights(ABC):  # noqa: B024
    def permute_incoming(self, sort_indices: torch.Tensor) -> None:
        raise NotImplementedError  # pragma: nocover

    def permute_outgoing(self, sort_indices: torch.Tensor) -> None:
        raise NotImplementedError  # pragma: nocover

    def calculate_outgoing_sort_order(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: nocover

    @property
    def weights(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: nocover

    def scale_down(self, scale: float) -> None:
        raise NotImplementedError  # pragma: nocover

    def set_biases_to_zero(self) -> None:
        raise NotImplementedError  # pragma: nocover
