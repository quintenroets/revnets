import pytest
import torch
from hypothesis import HealthCheck, given, settings

from revnets.evaluations.weights.standardize import align, standardize
from revnets.test_utils import test_utils

suppressed = (HealthCheck.function_scoped_fixture,)


@pytest.fixture()
def model():
    return test_utils.initialize_model()


@pytest.fixture()
def standardized_model():
    standardized_model = test_utils.initialize_model()
    standardize.standardize(standardized_model)
    return standardized_model


@given(inputs=test_utils.network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2, deadline=20000)
def test_align_preserves_behavior(model, standardized_model, inputs):
    align.align(model, standardized_model)
    inputs = test_utils.prepare_inputs(inputs)
    with torch.no_grad():
        outputs = (model(inputs), standardized_model(inputs))
    assert torch.allclose(*outputs, rtol=1e-3)
