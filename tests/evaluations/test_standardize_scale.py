import pytest
import torch
from hypothesis import HealthCheck, settings
from revnets.evaluations.weights.standardize import (
    calculate_incoming_weight_scales,
    standardize,
)
from revnets.networks import NetworkFactory, mininet

suppressed = (HealthCheck.function_scoped_fixture,)


model_factories = [
    mininet.NetworkFactory(),
]


@pytest.mark.parametrize("network_factory", model_factories)
def test_standardized_conditions(network_factory: NetworkFactory) -> None:
    network = network_factory.create_network()
    standardizer = standardize.Standardizer(network)
    standardizer.standardize_scale()
    for layer in standardizer.model_layers[:-1]:
        verify_standardized_scale(layer)


def verify_standardized_scale(layer: torch.nn.Module) -> None:
    scales = calculate_incoming_weight_scales(layer)
    ones = torch.ones_like(scales)
    close_to_one = torch.isclose(scales, ones)
    assert torch.all(close_to_one)


# @given(inputs=network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2, deadline=20000)
@pytest.mark.skip()
def test_standardize_preserves_functionality(network: torch.nn.Module, inputs) -> None:
    standardize.standardize_scale(network)
