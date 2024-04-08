import pytest
import torch
from hypothesis import HealthCheck, given, settings
from revnets.evaluations.weights.standardize import align, standardize

from tests.utils import initialize_model, network_inputs, prepare_inputs

suppressed = (HealthCheck.function_scoped_fixture,)


@pytest.fixture()
def model():
    return initialize_model()


@pytest.fixture()
def standardized_model():
    standardized_model = initialize_model()
    standardize.standardize(standardized_model)
    return standardized_model


@given(inputs=network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2, deadline=20000)
def test_align_preserves_behavior(model, standardized_model, inputs) -> None:
    align.align(model, standardized_model)
    inputs = prepare_inputs(inputs)
    with torch.no_grad():
        outputs = (model(inputs), standardized_model(inputs))
    assert torch.allclose(*outputs, rtol=1e-3)
