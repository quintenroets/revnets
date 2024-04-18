import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import Module

from .layer import InternalLayer
from .network import Standardizer
from .utils import extract_internal_layers


def align(model: Module, target: Module) -> None:
    Standardizer(model).standardize_scale()
    Standardizer(target).standardize_scale()
    # align internal layers of model to internal layers of target
    # to achieve a minimal weight difference
    layers = extract_internal_layers(model)
    target_layers = extract_internal_layers(target)
    for layer_pair in zip(layers, target_layers):
        align_layers(*layer_pair)


def align_layers(layer: InternalLayer, target: InternalLayer) -> None:
    sort_indices = calculate_optimal_order_mapping(
        layer.weights.weights, target.weights.weights
    )
    layer.weights.permute_outgoing(sort_indices)
    layer.next.permute_incoming(sort_indices)


def calculate_optimal_order_mapping(
    weights: torch.Tensor, target_weights: torch.Tensor
) -> torch.Tensor:
    distances = torch.cdist(target_weights, weights, p=1).numpy()
    indices = linear_sum_assignment(distances)[1]
    return torch.from_numpy(indices)
