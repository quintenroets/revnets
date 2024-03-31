import math

import torch
from hypothesis import strategies
from revnets.evaluations import weights
from revnets.networks import models
from revnets.utils import config

MAX_SIZE = 100


def get_input_size():
    model = initialize_model()
    return model.layer1.weight.shape[1]


def network_inputs():
    floats = strategies.floats(min_value=-10.0, max_value=10.0)
    input_size = get_input_size()
    size = min(input_size, MAX_SIZE)
    list_of_floats = strategies.lists(elements=floats, min_size=size, max_size=size)
    return list_of_floats


def initialize_model(**kwargs):
    torch.manual_seed(config.manual_seed)
    return models.mediumnet_images.Model(
        input_size=784, hidden_size1=100, hidden_size2=50, **kwargs
    )


def prepare_inputs(inputs: list[float]):
    deterministic_size = get_input_size() - MAX_SIZE
    inputs = inputs + [0] * deterministic_size
    return torch.Tensor(inputs).unsqueeze(0)


def are_isomorphism(model, model2, tanh: bool = False):
    """
    Check that models are different but equal up to isomorphism.
    """

    evaluator = weights.mse.Evaluator(model2, None, use_align=False, tanh=tanh)
    aligned_evaluator = weights.mse.Evaluator(model2, None, use_align=True, tanh=tanh)
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


def make_identical(model, model2) -> None:
    state = model.state_dict()
    model2.load_state_dict(state)
