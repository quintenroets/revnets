import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies

from revnets.evaluations.weights import standardize
from revnets.networks import mininet

suppressed = (HealthCheck.function_scoped_fixture,)


def network_inputs():
    INPUT_SIZE = 40
    floats = strategies.floats(min_value=-100.0, max_value=100.0)
    list_of_floats = strategies.lists(
        elements=floats, min_size=INPUT_SIZE, max_size=INPUT_SIZE
    )
    return list_of_floats


@pytest.fixture()
def model():
    return mininet.Network().get_architecture()


@pytest.fixture()
def standardized_model():
    standardized_model = mininet.Network().get_architecture()
    standardize(standardized_model)
    return standardized_model


@given(inputs=network_inputs())
@settings(suppress_health_check=suppressed, max_examples=5)
def test_standardize_preserves_behavior(model, standardized_model, inputs):
    with torch.no_grad():
        inputs = torch.Tensor(inputs)
        outputs = (model(inputs), standardized_model(inputs))
        assert torch.allclose(*outputs)
