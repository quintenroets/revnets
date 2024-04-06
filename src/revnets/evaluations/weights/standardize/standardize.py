from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import torch

from .internal_neurons import InternalNeurons


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
        model_layers = get_layers(model)
        pprint(model_layers)
        raise Exception
        return
        standardize_scale(model, tanh=tanh)
        raise NotImplementedError
        """For layers in zip(model_layers, model_layers[1:]):

        order.standardize_layers(*layers)
        """

    def standardize_scale(self) -> None:
        for neurons in self.generate_internal_neurons():
            neurons.standardize_scale()

        """
        if not tanh:
            # 2) optimize mae by distributing last layer scale factor over all layers
            out_scale = scale.get_scales(model_layers[-1])
            out_scale_total = sum(out_scale) / len(out_scale)
            avg_scale = out_scale_total ** (1 / len(model_layers))
            for layers in zip(model_layers, model_layers[1:]):
                scale.standardize_layers(*layers, scale=avg_scale, tanh=tanh)
        """

    def generate_internal_neurons(self) -> Iterator[InternalNeurons]:
        for incoming, outgoing in zip(self.model_layers, self.model_layers[1:]):
            yield InternalNeurons(incoming, outgoing)

    @cached_property
    def model_layers(self) -> list[torch.nn.Module]:
        """
        :return: all root layers (the deepest level) in order of feature propagation
        """
        layers = generate_layers(self.model)
        return list(layers)


def generate_layers(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    children = list(model.children())
    pprint(children)
    if children:
        for child in children:
            yield from generate_layers(child)
    else:
        yield model
