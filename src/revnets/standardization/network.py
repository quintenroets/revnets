from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, cast

from torch import nn

from revnets.networks.rnn import CreateRNNInput

from . import order, scale
from .internal_connection import InternalConnection

T = TypeVar("T")


@dataclass
class Standardizer:
    model: nn.Module
    optimize_mae: bool = False

    def run(self) -> None:
        """
        Convert network to the standard form of its isomorphism group.
        """
        self.standardize_scale()
        for connection in self.internal_connections:
            order.Standardizer(connection).run()
        if self.optimize_mae:
            self.apply_optimize_mae()

    def standardize_scale(self) -> None:
        for connection in self.internal_connections:
            scale.Standardizer(connection).run()

    def apply_optimize_mae(self) -> None:
        # optimize mae by distributing last layer scale factor over all layers
        can_optimize_mae = all(
            connection.has_norm_isomorphism for connection in self.internal_connections
        )
        if can_optimize_mae:
            self._apply_optimize_mae()

    def _apply_optimize_mae(self) -> None:
        desired_scale = self.calculate_average_scale_per_layer()
        for connection in self.internal_connections:
            connection.standardized_scale = desired_scale
            scale.Standardizer(connection).run()

    def calculate_average_scale_per_layer(self) -> float:
        connection = self.internal_connections[-1]
        standardizer = scale.Standardizer(connection)
        scales = standardizer.calculate_outgoing_scales(connection.output_weights)
        output_scale = sum(scales) / len(scales)
        num_internal_connections = len(self.internal_connections)
        average_scale = output_scale ** (1 / num_internal_connections)
        return cast(float, average_scale)

    @cached_property
    def internal_connections(self) -> list[InternalConnection]:
        return list(generate_internal_connections(self.model))


def generate_internal_connections(model: nn.Module) -> Iterator[InternalConnection]:
    layers = generate_layers(model)
    for triplet in generate_triplets(layers):
        yield InternalConnection(*triplet)


def generate_triplets(items: Iterable[T]) -> Iterator[tuple[T, T, T]]:
    items_list = list(items)
    yield from zip(items_list[::2], items_list[1::2], items_list[2::2])


# TODO: MaxPool destroys sign isomorphism for tanh
skip_layer_types = (
    nn.Flatten,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    CreateRNNInput,
)


def generate_layers(model: nn.Module) -> Iterator[nn.Module]:
    """
    :return: all root layers (the deepest level) in order of feature propagation
    """
    children = list(model.children())
    if children:
        for child in children:
            yield from generate_layers(child)
    else:
        if not isinstance(model, skip_layer_types):
            yield model
