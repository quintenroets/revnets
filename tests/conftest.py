import warnings

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from revnets.evaluations.weights.standardize import align, standardize

from tests.utils import initialize_model, network_inputs, prepare_inputs

warnings.filterwarnings("ignore", category=DeprecationWarning)
suppressed = (HealthCheck.function_scoped_fixture,)


@pytest.fixture()
def model():
    return initialize_model()


@pytest.fixture()
def standardized_model():
    standardized_model = initialize_model()
    standardize.standardize(standardized_model)
    return standardized_model
