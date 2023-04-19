import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies

from revnets import networks
from revnets.evaluations.weights.standardize import align, standardize

suppressed = (HealthCheck.function_scoped_fixture,)


def network_inputs():
    INPUT_SIZE = 40
    floats = strategies.floats(min_value=-100.0, max_value=100.0)
    list_of_floats = strategies.lists(
        elements=floats, min_size=INPUT_SIZE, max_size=INPUT_SIZE
    )
    return list_of_floats


def initialize_model():
    return networks.mediumnet.mediumnet_20.Network().get_architecture()


@pytest.fixture()
def model():
    return initialize_model()


@pytest.fixture()
def standardized_model():
    standardized_model = initialize_model()
    standardize.standardize(standardized_model)
    return standardized_model


@given(inputs=network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2)
def test_align_preserves_behavior(model, standardized_model, inputs):
    align.align(model, standardized_model)
    with torch.no_grad():
        inputs = torch.Tensor(inputs)
        outputs = (model(inputs), standardized_model(inputs))
        assert torch.allclose(*outputs, rtol=1e-3)
