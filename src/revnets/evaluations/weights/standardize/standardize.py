from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import torch

from .internal_neurons import InternalNeurons

from typing import TypeVar

T = TypeVar("T")


@dataclass
class Standardizer:
    model: torch.nn.Module

    def run(self) -> None:
        pass

    def standardize(self) -> None:
        """
        Convert network to the standard form of its isomorphism group.
        """
        self.standardize_scale()
        self.standardize_order()

    def standardize_order(self) -> None:
        raise NotImplementedError
        """For layers in zip(model_layers, model_layers[1:]):

        order.standardize_layers(*layers)
        """

    def standardize_scale(self) -> None:
        for neurons in self.internal_neurons:
            neurons.standardize_scale()

    def optimize_mae(self) -> None:
        # optimize mae by distributing last layer scale factor over all layers
        if all(neuron.has_norm_isomorphism for neuron in self.internal_neurons):
            scale = self.calculate_average_scale_per_layer()
            for neurons in self.internal_neurons:
                neurons.standardized_scale = scale
                neurons.standardize_scale()

    def calculate_average_scale_per_layer(self) -> float:
        last_neuron_scales = self.internal_neurons[-1]
        last_neuron_scale = sum(last_neuron_scales) / len(last_neuron_scales)
        return last_neuron_scale ** (1 / len(self.internal_neurons))

    @cached_property
    def internal_neurons(self) -> list[InternalNeurons]:
        neurons = self.generate_internal_neurons()
        return list(neurons)

    def generate_internal_neurons(self) -> Iterator[InternalNeurons]:
        for triplet in self.generate_triplets(self.model_layers):
            yield InternalNeurons(*triplet)

    @classmethod
    def generate_triplets(cls, items: list[T]) -> Iterator[tuple[T, T, T]]:
        yield from zip(items, items[1:], items[2:])

    @cached_property
    def model_layers(self) -> list[torch.nn.Module]:
        """
        :return: all root layers (the deepest level) in order of feature propagation
        """
        layers = generate_layers(self.model)
        return list(layers)


def generate_layers(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    children = list(model.children())
    if children:
        for child in children:
            yield from generate_layers(child)
    else:
        yield model
