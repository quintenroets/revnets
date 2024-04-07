from dataclasses import dataclass

import torch
from torch.nn import LeakyReLU, ReLU, Tanh


@dataclass
class InternalNeurons:
    incoming: torch.nn.Module
    activation: torch.nn.Module
    outgoing: torch.nn.Module
    standardized_scale: float = 1

    @property
    def has_norm_isomorphism(self) -> bool:
        activations = ReLU, LeakyReLU
        return isinstance(self.activation, activations)

    @property
    def has_sign_isomorphism(self) -> bool:
        return isinstance(self.activation, Tanh)

    @property
    def has_scale_isomorphism(self) -> bool:
        return self.has_norm_isomorphism or self.has_sign_isomorphism
