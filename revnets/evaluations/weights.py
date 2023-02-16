import torch


def evaluate(original: torch.nn.Module, reconstruction: torch.nn.Module, *_, **__):
    return get_mse(original, reconstruction)


def get_mse(original: torch.nn.Module, reconstruction: torch.nn.Module):
    standardize(original)
    standardize(reconstruction)

    MSE_size = sum(weights.numel() for weights in original.state_dict().values())
    MSE_sum = sum(
        torch.nn.functional.mse_loss(
            original_weights, reconstructed_weights, reduction="sum"
        ).item()
        for original_weights, reconstructed_weights in zip(
            original.state_dict().values(), reconstruction.state_dict().values()
        )
    )
    return MSE_sum / MSE_size


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


def get_output_sort_order(layer):
    weight = next(iter(layer.parameters()))
    total_output_weights = weight.norm(dim=1)
    _, output_sort_order = torch.sort(total_output_weights)
    return output_sort_order


def standardize(model: torch.nn.Module):
    """network weights are only defined up to isomorphisms, so standardize the
    weights before comparing.
    """
    layers = get_layers(model)
    for layer1, layer2 in zip(layers, layers[1:]):
        sort_indices = get_output_sort_order(layer1)
        permute_output_neurons(layer1, sort_indices)
        permute_input_neurons(layer2, sort_indices)


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
