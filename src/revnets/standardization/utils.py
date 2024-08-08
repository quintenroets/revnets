from collections.abc import Iterator
from itertools import tee

from torch import nn
from torch.nn.modules.pooling import _AvgPoolNd as AvgPoolND
from torch.nn.modules.pooling import _MaxPoolNd as MaxPoolND

from revnets.networks.images.rnn import CreateRNNInput, ExtractRNNOutput

from .layer import InternalLayer, Layer
from .scale_isomorphism import ScaleIsomorphism, determine_scale_isomorphism
from .weights import feedforward, rnn

skip_layers = nn.Flatten, AvgPoolND, CreateRNNInput, ExtractRNNOutput


def extract_internal_layers(network: nn.Module) -> Iterator[InternalLayer]:
    layers, next_layers = tee(extract_layers(network))
    next(next_layers, None)
    for layer, next_layer in zip(layers, next_layers, strict=False):
        yield InternalLayer(layer.weights, layer.scale_isomorphism, next_layer.weights)


def extract_layers(network: nn.Module) -> Iterator[Layer]:
    layers = extract_children(network)
    for layer in layers:
        if isinstance(layer, nn.RNN):
            yield from extract_rnn_layers(layer)
        else:
            yield extract_layer(layer, layers)


def extract_children(network: nn.Module) -> Iterator[nn.Module]:
    """
    :return: all root layers (the deepest level) in order of feature propagation
    """
    children = list(network.children())
    if children:
        for child in children:
            yield from extract_children(child)
    elif not isinstance(network, skip_layers):
        yield network


def extract_rnn_layers(layer: nn.RNN) -> Iterator[Layer]:
    for sub_layer in range(layer.num_layers):
        weights = rnn.extract_weights(layer, sub_layer)
        yield Layer(weights, scale_isomorphism=None)


def extract_layer(layer: nn.Module, layers: Iterator[nn.Module] | None = None) -> Layer:
    if layers is None:
        layers = iter([])
    weights = feedforward.extract_weights(layer)
    scale_isomorphism = extract_scale_isomorphism(layers)
    return Layer(weights, scale_isomorphism)


def extract_scale_isomorphism(layers: Iterator[nn.Module]) -> ScaleIsomorphism | None:
    activation = next(layers, None)
    return (
        extract_scale_isomorphism_after_max_pool(layers)
        if isinstance(activation, MaxPoolND)
        else determine_scale_isomorphism(activation)
    )


def extract_scale_isomorphism_after_max_pool(
    layers: Iterator[nn.Module],
) -> ScaleIsomorphism | None:
    activation = next(layers, None)
    isomorphism = determine_scale_isomorphism(activation)
    if isomorphism is ScaleIsomorphism.sign:
        isomorphism = None
    return isomorphism
