import torch

from . import order


def standardize_layers(layer1, layer2):
    """Standardize by multiplying incoming weights and biases by scale and
    outgoing weights with the inverse scale.
    """
    scales = get_scales(layer1, layer2)
    rescale_incoming_weights(layer1, 1 / scales)
    rescale_outgoing_weights(layer2, scales)


def get_scales(layer, layer2):
    weights_in = order.get_layer_weights(layer)
    scales_in = weights_in.norm(dim=1)

    weights_out = next(iter(layer2.parameters()))

    scales_out = weights_out.norm(dim=0)
    scales_out[scales_out == 0] = 1

    scales = torch.sqrt(scales_in / scales_out)
    # we want to make the norm of the ingoing and the outgoing weights equal

    scales[scales == 0] = 1
    return scales


def rescale_incoming_weights(layer, scales):
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer, scales):
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
