from . import order


def standardize_layers(layer1, layer2, scale=None):
    """Standardize by multiplying incoming weights and biases by scale and
    outgoing weights with the inverse scale.
    """
    scales = get_scales(layer1)
    if scale is not None:
        scales /= scale
    rescale_incoming_weights(layer1, 1 / scales)
    rescale_outgoing_weights(layer2, scales)


def get_scales(layer):
    weights = order.get_layer_weights(layer)
    scales = weights.norm(dim=1, p=2)
    return scales


def rescale_incoming_weights(layer, scales):
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer, scales):
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
