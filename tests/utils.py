import math

import torch
from revnets import networks
from revnets.evaluations import weights
from torch.nn import Sequential


def extract_input_size(network: Sequential) -> int:
    layers = network.children()
    return next(layers).weight.shape[1]


def create_network_inputs(network: Sequential) -> torch.Tensor:
    size = 1, extract_input_size(network)
    return torch.rand(size) * 20 - 10


def initialize_model(**kwargs) -> Sequential:
    # torch.manual_seed(config.manual_seed)
    return networks.mediumnet_images.NetworkFactory(
        input_size=784, hidden_size1=100, hidden_size2=50, **kwargs
    ).create_network()


def are_isomorphism(model, model2):
    """
    Check that models are different but equal up to isomorphism.
    """

    evaluator = weights.mse.Evaluator(model2, None)
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


def make_identical(model, model2) -> None:
    state = model.state_dict()
    model2.load_state_dict(state)
