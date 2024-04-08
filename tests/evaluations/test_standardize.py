from types import ModuleType

import pytest
import torch
from revnets.evaluations.weights import standardize
from revnets.evaluations.weights.standardize import Standardizer
from revnets.models import InternalNeurons
from torch import nn
from torch.nn import Sequential

from tests import utils
from tests.utils import (
    activation_layers,
    create_network,
    network_modules,
    only_scale_options,
)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("only_scale", only_scale_options)
def test_standardized_form(
    network_module: ModuleType, activation_layer: nn.Module, only_scale: bool
) -> None:
    network = create_network(network_module, activation_layer)
    standardizer = Standardizer(network)
    if only_scale:
        standardizer.standardize_scale()
    else:
        standardizer.run()
    verify_standardized_form(network, only_scale)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("only_scale", only_scale_options)
def test_standardize_preserves_functionality(
    network_module: ModuleType, activation_layer: nn.Module, only_scale: bool
) -> None:
    network = create_network(network_module, activation_layer)
    verify_functional_preservation(network, only_scale)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("only_scale", only_scale_options)
def test_standardized_form_and_functionality_preservation(
    network_module: ModuleType, activation_layer: nn.Module, only_scale: bool
) -> None:
    network = create_network(network_module, activation_layer)
    verify_functional_preservation(network, only_scale)
    verify_standardized_form(network, only_scale)


def verify_functional_preservation(network: Sequential, only_scale: bool) -> None:
    standardizer = Standardizer(network)
    transformation = standardizer.standardize_scale if only_scale else standardizer.run
    utils.verify_functional_preservation(network, transformation)


def verify_standardized_form(network: Sequential, only_scale: bool) -> None:
    neuron_layers = standardize.generate_internal_neurons(network)
    for neurons in neuron_layers:
        if neurons.has_scale_isomorphism:
            verify_scale_standardized(neurons)
        if not only_scale:
            verify_order_standardized(neurons)


def verify_scale_standardized(neurons: InternalNeurons) -> None:
    neurons_standardizer = standardize.scale.Standardizer(neurons)
    scales = neurons_standardizer.calculate_scale_factors(neurons.incoming)
    ones = torch.ones_like(scales)
    close_to_one = torch.isclose(scales, ones)
    assert torch.all(close_to_one)


def verify_order_standardized(neurons: InternalNeurons) -> None:
    weights = standardize.extract_layer_weights(neurons.incoming)
    incoming_weights = weights.norm(dim=1, p=1)
    sorted_indices = incoming_weights[:-1] <= incoming_weights[1:]
    is_sorted = torch.all(sorted_indices)
    assert is_sorted
