import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings

from revnets.evaluations import weights
from revnets.evaluations.weights.standardize import standardize
from revnets.test_utils import test_utils

suppressed = (HealthCheck.function_scoped_fixture,)


@pytest.fixture()
def model():
    return test_utils.initialize_model()


@pytest.fixture()
def model2():
    return test_utils.initialize_model()


@pytest.fixture()
def standardized_model():
    standardized_model = test_utils.initialize_model()
    standardize.standardize(standardized_model)
    return standardized_model


@given(inputs=test_utils.network_inputs())
@settings(suppress_health_check=suppressed, max_examples=2)
def test_standardize_preserves_behavior(model, standardized_model, inputs):
    inputs = test_utils.prepare_inputs(inputs)
    with torch.no_grad():
        outputs = (model(inputs), standardized_model(inputs))
    assert torch.allclose(*outputs, rtol=1e-3)


def test_permutation_standardization(model, model2):
    make_identical(model, model2)

    model_layers = standardize.get_layers(model)
    model2_layers = standardize.get_layers(model2)
    layer_iterator = model_layers, model_layers[1:], model2_layers, model2_layers[1:]
    order = standardize.order
    for iteration in zip(*layer_iterator):
        model_layer1, model_layer2, model2_layer1, model2_layer2 = iteration  # noqa
        sort_indices = order.get_output_sort_order(model_layer1)
        order.permute_output_neurons(model_layer1, sort_indices)
        order.permute_input_neurons(model_layer2, sort_indices)

        sort_indices_reverse = torch.flip(sort_indices, (0,))
        order.permute_output_neurons(model2_layer1, sort_indices_reverse)
        order.permute_input_neurons(model2_layer2, sort_indices_reverse)

    assert are_isomorphism(model, model2)


def test_weight_standardization(model, model2):
    make_identical(model, model2)

    model_layers = standardize.get_layers(model)
    for layer1, layer2 in zip(model_layers, model_layers[1:]):
        scales = standardize.scale.get_scales(layer1)
        wrong_scales = torch.ones_like(scales) * 2
        standardize.scale.rescale_incoming_weights(layer1, 1 / wrong_scales)
        standardize.scale.rescale_outgoing_weights(layer2, wrong_scales)

    assert are_isomorphism(model, model2)


def are_isomorphism(model, model2):
    """Check that models are different but equal up to isomorphism."""

    evaluator = weights.mse.Evaluator(model2, None, use_align=False)
    aligned_evaluator = weights.mse.Evaluator(model2, None, use_align=True)
    for evaluator_used in evaluator, aligned_evaluator:
        evaluator_used.original = model.to(evaluator.device)

    mse = evaluator.calculate_distance()
    standardized_mse = evaluator.evaluate()
    aligned_mse = aligned_evaluator.evaluate()
    tol = 1e-5
    are_different = not math.isclose(mse, 0, abs_tol=tol)
    are_identical_after_standardization = math.isclose(standardized_mse, 0, abs_tol=tol)
    are_identical_after_alignment = math.isclose(aligned_mse, 0, abs_tol=tol)
    return (
        are_different
        and are_identical_after_standardization
        and are_identical_after_alignment
    )


def make_identical(model, model2):
    state = model.state_dict()
    model2.load_state_dict(state)
