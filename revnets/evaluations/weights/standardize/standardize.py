import torch

from revnets.utils.config import Activation, config

from . import order, scale


def standardize(model: torch.nn.Module, tanh: bool = None):
    """network weights are only defined up to isomorphisms, so standardize the
    weights before comparing.
    """
    model_layers = get_layers(model)
    standardize_scale(model, tanh=tanh)
    for layers in zip(model_layers, model_layers[1:]):
        order.standardize_layers(*layers)


def standardize_scale(model: torch.nn.Module, tanh: bool = None):
    if tanh is None:
        tanh = config.activation == Activation.tanh

    model_layers = get_layers(model)

    # 1) standardize
    for layers in zip(model_layers, model_layers[1:]):
        scale.standardize_layers(*layers, tanh=tanh)

    if not tanh:
        # 2) optimize mae by distributing last layer scale factor over all layers
        out_scale = scale.get_scales(model_layers[-1])
        out_scale_total = sum(out_scale) / len(out_scale)
        avg_scale = out_scale_total ** (1 / len(model_layers))
        for layers in zip(model_layers, model_layers[1:]):
            scale.standardize_layers(*layers, scale=avg_scale, tanh=tanh)


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
