from types import ModuleType

import pytest
from revnets import networks
from torch import nn

from tests.evaluations.verifier import StandardizationType, Verifier

standardization_types = (
    StandardizationType.scale,
    StandardizationType.standardize,
    StandardizationType.align,
)
network_modules = (networks.mininet,)
activation_layers = (
    nn.ReLU(),
    nn.LeakyReLU(),
    nn.Tanh(),
)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("standardization_type", standardization_types)
def test_standardized_form(
    network_module: ModuleType,
    activation_layer: nn.Module,
    standardization_type: StandardizationType,
) -> None:
    tester = Verifier(network_module, activation_layer, standardization_type)
    tester.test_standardized_form()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("standardization_type", standardization_types)
def test_standardize_preserves_functionality(
    network_module: ModuleType,
    activation_layer: nn.Module,
    standardization_type: StandardizationType,
) -> None:
    tester = Verifier(network_module, activation_layer, standardization_type)
    tester.test_functional_preservation()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("standardization_type", standardization_types)
def test_standardized_form_and_functionality_preservation(
    network_module: ModuleType,
    activation_layer: nn.Module,
    standardization_type: StandardizationType,
) -> None:
    tester = Verifier(network_module, activation_layer, standardization_type)
    tester.test_functional_preservation()
    tester.test_standardized_form()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation_layer", activation_layers)
@pytest.mark.parametrize("standardization_type", standardization_types)
def test_second_standardize_no_effect(
    network_module: ModuleType,
    activation_layer: nn.Module,
    standardization_type: StandardizationType,
) -> None:
    tester = Verifier(network_module, activation_layer, standardization_type)
    tester.test_second_standardize_no_effect()
