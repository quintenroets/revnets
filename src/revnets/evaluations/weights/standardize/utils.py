import torch
from torch.nn import Module


def extract_layer_weights(
    layer: Module, device: torch.device | None = None
) -> torch.Tensor:
    with torch.no_grad():
        connection_weights, bias_weights = layer.parameters()
        weights_tuple = (connection_weights, bias_weights.reshape(-1, 1))
        weights = torch.hstack(weights_tuple)
    if device is not None:
        weights = weights.to(device)
    return weights
