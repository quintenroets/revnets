import torch

from . import order, scale


def standardize(model: torch.nn.Module):
    """network weights are only defined up to isomorphisms, so standardize the
    weights before comparing.
    """
    model_layers = get_layers(model)
    for layers in zip(model_layers, model_layers[1:]):
        standardize_layers(*layers)


def standardize_layers(*layers):
    order.standardize_layers(*layers)
    scale.standardize_layers(*layers)


def get_layers(model: torch.nn.Module):
    """
    :param model: neural network
    :return: list of all root layers (the deepest level) in order of feature propagation
    """
    children = list(model.children())
    layers = (
        [layer for child in children for layer in get_layers(child)]
        if children
        else [model]
    )
    return layers
