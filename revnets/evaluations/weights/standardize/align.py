import bisect

import torch

from . import order, standardize


def align(model: torch.nn.Module, model_to_align: torch.nn.Module):
    standardize.standardize_scale(model)
    standardize.standardize_scale(model_to_align)

    # align neurons of model_to_align to neurons of model
    # to achieve a minimal weight difference

    model_layers = standardize.get_layers(model)
    model_layers_to_align = standardize.get_layers(model_to_align)

    layer_pairs = zip(model_layers, model_layers[1:])
    layer_pairs_to_align = zip(model_layers_to_align, model_layers_to_align[1:])

    for layer_pair, layer_pair_to_align in zip(layer_pairs, layer_pairs_to_align):
        # for layers in (layer_pair, layer_pair_to_align):
        # scale.standardize_layers(*layers)

        align_layers(layer_pair, layer_pair_to_align)


def align_layers(layer_pair, layer_pair_to_align):
    sort_indices = get_align_order(layer_pair[0], layer_pair_to_align[0])
    order.permute_output_neurons(layer_pair_to_align[0], sort_indices)
    order.permute_input_neurons(layer_pair_to_align[1], sort_indices)


def get_align_order(layer, layer_to_align):
    weights = order.get_layer_weights(layer)
    weights_to_align = order.get_layer_weights(layer_to_align)

    match_pairs = []
    while len(match_pairs) < len(weights):
        pair = find_match_pair(match_pairs, weights, weights_to_align)
        match_pairs.append(pair)

    matched_indices = torch.LongTensor([pair[0] for pair in match_pairs])
    matched_indices_to_align = torch.LongTensor([pair[1] for pair in match_pairs])
    sorted_match_indices, sort_order = torch.sort(matched_indices)

    return matched_indices_to_align[sort_order]


def find_match_pair(match_pairs, weights, weights_to_align):
    matched_indices = [pair[0] for pair in match_pairs]
    matched_indices_to_align = [pair[1] for pair in match_pairs]

    min_distance = None
    min_index = None
    min_index_orig = None
    for i, weight in enumerate(weights):
        if i not in matched_indices:
            matching_index, distance = get_most_similar_index(
                weight, weights_to_align, matched_indices_to_align
            )
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_index = matching_index
                min_index_orig = i

    return min_index_orig, min_index


def get_most_similar_index(weight, weights, matched_indices_to_align):
    distances = weights - weight
    # use l1-norm because l2-norm is already standardized
    distances = distances.norm(dim=1, p=1)
    min_distance = None
    min_index = None
    for i, distance in enumerate(distances):
        if i not in matched_indices_to_align:
            if min_distance is None or distance < min_distance:
                min_index = i
                min_distance = distance
    return min_index, min_distance


def get_align_order_old(layer, layer_to_align):
    weights = order.get_layer_weights(layer)
    weights_to_align = order.get_layer_weights(layer_to_align)

    matching_indices = []
    sorted_matching_indices = []

    for weight in weights:
        matching_index = get_most_similar_index_old(weight, weights_to_align)
        # remove index from future matching
        weights_to_align = remove_index(weights_to_align, matching_index)
        matching_index = get_original_index(sorted_matching_indices, matching_index)
        matching_indices.append(matching_index)
        bisect.insort(sorted_matching_indices, matching_index)

    return torch.LongTensor(matching_indices)


def get_most_similar_index_old(weight, weights):
    distances = weights - weight
    # use l2-norm because l1-norm is already standardized
    distances = distances.norm(dim=1, p=1)
    return torch.argmin(distances).item()


def remove_index(weights, index):
    return torch.vstack((weights[:index, :], weights[index + 1 :, :]))


def get_original_index(sorted_indices, index):
    # calculate index in original weights if no indices were removed
    for previous_index in sorted_indices:
        if previous_index <= index:
            index += 1
    return index
