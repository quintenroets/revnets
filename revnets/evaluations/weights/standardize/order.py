import torch


def standardize_layers(layer1, layer2):
    sort_indices = get_output_sort_order(layer1)
    permute_output_neurons(layer1, sort_indices)
    permute_input_neurons(layer2, sort_indices)


def get_output_sort_order(layer):
    weights = get_layer_weights(layer)
    total_output_weights = weights.norm(dim=1, p=1)
    # use l1-norm because l2-norm is already standardized
    _, output_sort_order = torch.sort(total_output_weights)
    return output_sort_order


def permute_input_neurons(layer, sort_indices):
    length = len(sort_indices)
    for param in layer.parameters():
        shape = param.data.shape
        if len(shape) == 2 and shape[1] == length:
            param.data = param.data[:, sort_indices]


def permute_output_neurons(layer, sort_indices):
    length = len(sort_indices)
    for param in layer.parameters():
        shape = param.shape
        dims = len(shape)
        if dims in (1, 2) and shape[0] == length:
            param.data = (
                param.data[sort_indices] if dims == 1 else param.data[sort_indices, :]
            )


def get_layer_weights(layer, device=None) -> torch.Tensor:
    with torch.no_grad():
        connection_weights, bias_weights = layer.parameters()
        weights = (connection_weights, bias_weights.reshape(-1, 1))
        weights = torch.hstack(weights)
        if device is not None:
            weights = weights.to(device)
        return weights
