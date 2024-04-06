import pytest
from types import ModuleType
from torch.nn import Sequential
import torch
from torch import nn
from revnets.evaluations.weights.standardize import standardize
from revnets.networks import mininet
from revnets.evaluations.weights.standardize.standardize import Standardizer
from tests.utils import create_network_inputs

activation_layers = nn.ReLU(), nn.LeakyReLU(), nn.Tanh()

network_modules = (mininet,)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
def test_standardized_form(
    network_module: ModuleType, activation_layer: nn.Module
) -> None:
    network_factory = network_module.NetworkFactory(activation_layer=activation_layer)
    network = network_factory.create_network()
    standardize.Standardizer(network).standardize_scale()
    verify_standardized_form(network)


def verify_standardized_form(network: Sequential) -> None:
    standardizer = Standardizer(network)
    if standardizer.internal_neurons[0].has_scale_isomorphism:
        for neurons in standardizer.internal_neurons:
            scales = neurons.calculate_scale_factors(neurons.incoming)
            ones = torch.ones_like(scales)
            close_to_one = torch.isclose(scales, ones)
            assert torch.all(close_to_one)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
def test_standardize_preserves_functionality(
    network_module: ModuleType, activation_layer: nn.Module
) -> None:
    network_factory = network_module.NetworkFactory(activation_layer=activation_layer)
    network = network_factory.create_network()
    verify_functional_preservation(network)


def verify_functional_preservation(network: Sequential) -> None:
    inputs = create_network_inputs(network)
    outputs = network(inputs)
    standardize.Standardizer(network).standardize_scale()
    outputs_after_standardization = network(inputs)
    outputs_are_closes = torch.isclose(outputs, outputs_after_standardization)
    assert torch.all(outputs_are_closes)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
def test_standardized_form_and_functionality_preservation(
    network_module: ModuleType, activation_layer: nn.Module
) -> None:
    network_factory = network_module.NetworkFactory(activation_layer=activation_layer)
    network = network_factory.create_network()
    verify_functional_preservation(network)
    verify_standardized_form(network)
