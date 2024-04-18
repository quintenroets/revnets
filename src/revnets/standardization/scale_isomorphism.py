from collections.abc import Iterator
from enum import Enum

from torch import nn


class ScaleIsomorphism(Enum):
    sign = "sign"
    norm = "norm"


activations_per_isomorphism: dict[ScaleIsomorphism, tuple[type[nn.Module], ...]] = {
    ScaleIsomorphism.norm: (nn.ReLU, nn.LeakyReLU),
    ScaleIsomorphism.sign: (nn.Tanh,),
}


def determine_scale_isomorphism(
    activation: nn.Module | None,
) -> ScaleIsomorphism | None:
    return next(generate_scale_isomorphism(activation), None)


def generate_scale_isomorphism(
    activation: nn.Module | None,
) -> Iterator[ScaleIsomorphism]:
    for isomorphism, activations in activations_per_isomorphism.items():
        if isinstance(activation, activations):
            yield isomorphism
