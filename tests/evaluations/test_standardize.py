from types import ModuleType

import pytest
from revnets import networks
from revnets.models import Activation

from tests.evaluations.verifier import Standardization, Verifier

standardizations = (
    Standardization.scale,
    Standardization.standardize,
    Standardization.align,
)
network_modules = (
    networks.mininet,
    networks.mediumnet,
    networks.images.mininet,
    networks.images.mediumnet,
    networks.images.cnn.mini,
    networks.images.cnn.lenet,
)
activations = (
    Activation.leaky_relu,
    Activation.relu,
    Activation.tanh,
)


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("standardization_type", standardizations)
def test_standardized_form(
    network_module: ModuleType,
    activation: Activation,
    standardization_type: Standardization,
) -> None:
    tester = Verifier(network_module, activation, standardization_type)
    tester.test_standardized_form()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("standardization_type", standardizations)
def test_standardize_preserves_functionality(
    network_module: ModuleType,
    activation: Activation,
    standardization_type: Standardization,
) -> None:
    tester = Verifier(network_module, activation, standardization_type)
    tester.test_functional_preservation()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("standardization_type", standardizations)
def test_standardized_form_and_functionality_preservation(
    network_module: ModuleType,
    activation: Activation,
    standardization_type: Standardization,
) -> None:
    tester = Verifier(network_module, activation, standardization_type)
    tester.test_functional_preservation()
    tester.test_standardized_form()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("standardization_type", standardizations)
def test_second_standardize_no_effect(
    network_module: ModuleType,
    activation: Activation,
    standardization_type: Standardization,
) -> None:
    tester = Verifier(network_module, activation, standardization_type)
    tester.test_second_standardize_no_effect()


@pytest.mark.parametrize("network_module", network_modules)
@pytest.mark.parametrize("activation", activations)
def test_optimize_mae(network_module: ModuleType, activation: Activation) -> None:
    tester = Verifier(network_module, activation, Standardization.standardize)
    tester.test_optimize_mae()
