import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import Module

from revnets.models import InternalNeurons

from . import order
from .network import Standardizer, generate_internal_neurons


def align(model: Module, target: Module) -> None:
    Standardizer(model).standardize_scale()
    Standardizer(target).standardize_scale()
    # align internal neurons of model to internal neurons of target
    # to achieve a minimal weight difference
    neurons = generate_internal_neurons(model)
    target_neurons = generate_internal_neurons(target)
    for internal_neurons_pair in zip(neurons, target_neurons):
        align_internal_neurons(*internal_neurons_pair)


def align_internal_neurons(neurons: InternalNeurons, target: InternalNeurons) -> None:
    sort_indices = calculate_optimal_order(neurons.incoming, target.incoming)
    order.permute_output_weights(neurons.incoming, sort_indices)
    order.permute_input_weights(neurons.outgoing, sort_indices)


def calculate_optimal_order(layer: Module, target: Module) -> torch.Tensor:
    weights = order.extract_linear_layer_weights(layer)
    target_weights = order.extract_linear_layer_weights(target)
    distances = torch.cdist(target_weights, weights, p=1).numpy()
    indices = linear_sum_assignment(distances)[1]
    return torch.from_numpy(indices)
