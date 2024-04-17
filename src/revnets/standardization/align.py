import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import Module

from . import order
from .internal_connection import InternalConnection
from .network import Standardizer, generate_internal_connections


def align(model: Module, target: Module) -> None:
    Standardizer(model).standardize_scale()
    Standardizer(target).standardize_scale()
    # align internal neurons of model to internal neurons of target
    # to achieve a minimal weight difference
    connections = generate_internal_connections(model)
    target_connections = generate_internal_connections(target)
    for connection_pair in zip(connections, target_connections):
        align_internal_connections(*connection_pair)


def align_internal_connections(
    connection: InternalConnection, target: InternalConnection
) -> None:
    sort_indices = calculate_optimal_order_mapping(
        connection.input_weights, target.input_weights
    )
    order.permute_outgoing(connection.input_parameters, sort_indices)
    order.permute_incoming(connection.output_parameters, sort_indices)


def calculate_optimal_order_mapping(
    weights: torch.Tensor, target_weights: torch.Tensor
) -> torch.Tensor:
    distances = torch.cdist(target_weights, weights, p=1).numpy()
    indices = linear_sum_assignment(distances)[1]
    return torch.from_numpy(indices)
