from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from types import ModuleType
from typing import cast

import torch
from revnets.evaluations.weights import standardize
from revnets.evaluations.weights.standardize import Standardizer
from revnets.models import InternalNeurons
from torch import nn
from torch.nn import Sequential


class StandardizationType(Enum):
    scale = "scale"
    standardize = "standardize"
    align = "align"


@dataclass
class Verifier:
    network_module: ModuleType
    activation_layer: nn.Module
    standardization_type: StandardizationType

    def __post_init__(self) -> None:
        self.network = self.create_network()

    @cached_property
    def target(self) -> Sequential:
        return self.create_network()

    def test_standardized_form(self) -> None:
        self.apply_transformation()
        if self.standardization_type == StandardizationType.align:
            self.verify_aligned_form()
        else:
            self.verify_standardized_form()

    def apply_transformation(self) -> None:
        match self.standardization_type:
            case StandardizationType.scale:
                Standardizer(self.network).standardize_scale()
            case StandardizationType.standardize:
                return Standardizer(self.network).run()
            case StandardizationType.align:
                return standardize.align(self.network, self.target)

    def test_functional_preservation(self) -> None:
        inputs = self.create_network_inputs()
        with torch.no_grad():
            outputs = self.network(inputs)
        self.apply_transformation()
        with torch.no_grad():
            outputs_after_transformation = self.network(inputs)
        outputs_are_closes = torch.isclose(
            outputs, outputs_after_transformation, rtol=1e-3
        )
        assert torch.all(outputs_are_closes)

    def verify_standardized_form(self) -> None:
        neuron_layers = standardize.generate_internal_neurons(self.network)
        for neurons in neuron_layers:
            if neurons.has_scale_isomorphism:
                verify_scale_standardized(neurons)
            if self.standardization_type == StandardizationType.standardize:
                verify_order_standardized(neurons)

    def verify_aligned_form(self) -> None:
        neuron_layers = standardize.generate_internal_neurons(self.network)
        target_neuron_layers = standardize.generate_internal_neurons(self.target)
        for neurons, target_neurons in zip(neuron_layers, target_neuron_layers):
            if neurons.has_scale_isomorphism:
                verify_scale_standardized(neurons)
                verify_scale_standardized(target_neurons)
                verify_aligned(neurons, target_neurons)

    def create_network(self) -> Sequential:
        Factory = self.network_module.NetworkFactory
        factory = Factory(activation_layer=self.activation_layer)
        network = factory.create_network()
        return cast(Sequential, network)

    def create_network_inputs(self) -> torch.Tensor:
        size = 1, self.extract_input_size()
        return torch.rand(size) * 20 - 10

    def extract_input_size(self) -> int:
        layers = self.network.children()
        size = next(layers).weight.shape[1]
        return cast(int, size)

    def test_second_standardize_no_effect(self) -> None:
        self.apply_transformation()
        state_dict = self.network.state_dict()
        self.apply_transformation()
        second_state_dict = self.network.state_dict()
        for value, second_value in zip(state_dict.values(), second_state_dict.values()):
            is_close = torch.isclose(value, second_value)
            assert torch.all(is_close)


def verify_scale_standardized(neurons: InternalNeurons) -> None:
    neurons_standardizer = standardize.scale.Standardizer(neurons)
    scales = neurons_standardizer.calculate_scale_factors(neurons.incoming)
    ones = torch.ones_like(scales)
    close_to_one = torch.isclose(scales, ones)
    assert torch.all(close_to_one)


def verify_order_standardized(neurons: InternalNeurons) -> None:
    weights = standardize.extract_linear_layer_weights(neurons.incoming)
    incoming_weights = weights.norm(dim=1, p=1)
    sorted_indices = incoming_weights[:-1] <= incoming_weights[1:]
    is_sorted = torch.all(sorted_indices)
    assert is_sorted


def verify_aligned(neurons: InternalNeurons, target_neurons: InternalNeurons) -> None:
    order = standardize.calculate_optimal_order(
        neurons.incoming, target_neurons.incoming
    )
    is_ordered = order == torch.arange(len(order))
    assert torch.all(is_ordered)
