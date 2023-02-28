import torch


def standardize_layers(layer1, layer2):
    sort_indices = get_output_sort_order(layer1)
    permute_output_neurons(layer1, sort_indices)
    permute_input_neurons(layer2, sort_indices)


def get_output_sort_order(layer):
    weight = next(iter(layer.parameters()))
    total_output_weights = weight.norm(dim=1)
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
