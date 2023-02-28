import torch


def standardize_layers(layer1, layer2):
    """Standardize by multiplying incoming weights and biases by scale and
    outgoing weights with the inverse scale.
    """
    scales = get_output_scales(layer1)
    rescale_incoming_weights(layer1, 1 / scales)
    rescale_outgoing_weights(layer2, scales)


def get_output_scales(layer):
    weight = next(iter(layer.parameters()))
    scales = weight.norm(dim=1)
    scales[scales == 0] = 1
    scales = torch.abs(scales)
    return scales


def rescale_incoming_weights(layer, scales):
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer, scales):
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
