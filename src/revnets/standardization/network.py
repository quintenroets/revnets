from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, cast

from torch import nn

from revnets.models import InternalNeurons

from . import order, scale
from .utils import extract_weights

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
        for neurons in self.internal_neurons:
            order.Standardizer(neurons).run()
        if self.optimize_mae:
            self.apply_optimize_mae()

    def standardize_scale(self) -> None:
        for neurons in self.internal_neurons:
            scale.Standardizer(neurons).run()

    def apply_optimize_mae(self) -> None:
        # optimize mae by distributing last layer scale factor over all layers
        if all(neuron.has_norm_isomorphism for neuron in self.internal_neurons):
            desired_scale = self.calculate_average_scale_per_layer()
            for neurons in self.internal_neurons:
                neurons.standardized_scale = desired_scale
                scale.Standardizer(neurons).run()

    def calculate_average_scale_per_layer(self) -> float:
        weights = extract_weights(self.internal_neurons[-1].outgoing)
        last_neuron_scales = weights.norm(dim=1, p=2)
        last_neuron_scale = sum(last_neuron_scales) / len(last_neuron_scales)
        num_internal_layers = len(self.internal_neurons)
        average_scale = last_neuron_scale ** (1 / num_internal_layers)
        return cast(float, average_scale)

    @cached_property
    def internal_neurons(self) -> list[InternalNeurons]:
        neurons = generate_internal_neurons(self.model)
        return list(neurons)


def generate_internal_neurons(model: nn.Module) -> Iterator[InternalNeurons]:
    layers = generate_layers(model)
    layers_list = list(layers)
    for triplet in generate_triplets(layers_list):
        yield InternalNeurons(*triplet)


def generate_triplets(items: list[T]) -> Iterator[tuple[T, T, T]]:
    yield from zip(items[::2], items[1::2], items[2::2])


# TODO: MaxPool destroys sign isomorphism for tanh
skip_layer_types = nn.Flatten, nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d


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
