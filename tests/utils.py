import math
from collections.abc import Callable
from types import ModuleType

import torch
from revnets.evaluations import weights
from revnets.networks import mininet
from torch import nn
from torch.nn import Sequential

network_modules = (mininet,)

activation_layers = nn.ReLU(), nn.LeakyReLU(), nn.Tanh()
only_scale_options = True, False


def extract_input_size(network: Sequential) -> int:
    layers = network.children()
    return next(layers).weight.shape[1]


def create_network_inputs(network: Sequential) -> torch.Tensor:
    size = 1, extract_input_size(network)
    return torch.rand(size) * 20 - 10


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


def verify_functional_preservation(
    network: Sequential, transformation_callback: Callable[[], None]
) -> None:
    inputs = create_network_inputs(network)
    with torch.no_grad():
        outputs = network(inputs)
    transformation_callback()
    with torch.no_grad():
        outputs_after_transformation = network(inputs)
    outputs_are_closes = torch.isclose(outputs, outputs_after_transformation, rtol=1e-3)
    assert torch.all(outputs_are_closes)


def create_network(
    network_module: ModuleType, activation_layer: nn.Module
) -> Sequential:
    network_factory = network_module.NetworkFactory(activation_layer=activation_layer)
    return network_factory.create_network()
