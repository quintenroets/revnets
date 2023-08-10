import pytest
import torch
from hypothesis import HealthCheck, given, settings
from rich.pretty import pprint

from revnets.evaluations.weights.standardize import align, standardize
from revnets.test_utils import test_utils
from revnets.utils.config import Activation

suppressed = (HealthCheck.function_scoped_fixture,)


@pytest.fixture()
def model():
    return test_utils.initialize_model(activation=Activation.tanh)


@pytest.fixture()
def standardized_model():
    standardized_model = test_utils.initialize_model(activation=Activation.tanh)
    standardize.standardize(standardized_model, tanh=True)
    return standardized_model


@given(inputs=test_utils.network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2, deadline=20000)
def test_align_preserves_behavior(model, standardized_model, inputs):
    align.align(model, standardized_model, tanh=True)
    inputs = test_utils.prepare_inputs(inputs)
    with torch.no_grad():
        outputs = (model(inputs), standardized_model(inputs))
    assert torch.allclose(*outputs, rtol=1e-3)


if __name__ == "__main__":
    test_model = test_utils.initialize_model(activation=Activation.tanh)
    std_model = test_utils.initialize_model(activation=Activation.tanh)
    standardize.standardize(std_model, tanh=True)
    align.align(test_model, std_model, tanh=True)
    inputs = torch.randn(1, 784)
    a = test_model(inputs)
    b = std_model(inputs)
    assert torch.allclose(a, b, rtol=1e-3)
    pprint(a)
    pprint(b)
