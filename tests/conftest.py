import warnings

import pytest
from hypothesis import HealthCheck
from revnets.evaluations.weights.standardize import standardize

from tests.utils import initialize_model

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
