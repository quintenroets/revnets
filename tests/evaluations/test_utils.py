from typing import cast

from hypothesis import given, strategies
from revnets.standardization.utils import (
    extract_layer,
    extract_rnn_layers,
    extract_scale_isomorphism_after_max_pool,
)
from revnets.standardization.weights import rnn
from torch import nn


@given(
    in_features=strategies.integers(min_value=1, max_value=10),
    out_features=strategies.integers(min_value=1, max_value=10),
)
def test_linear_extract_weights(in_features: int, out_features: int) -> None:
    layer = nn.Linear(in_features=in_features, out_features=out_features)
    weights = extract_layer(layer).weights.weights
    assert weights.shape == (out_features, in_features + 1)


@given(
    in_channels=strategies.integers(min_value=1, max_value=10),
    out_channels=strategies.integers(min_value=1, max_value=10),
    kernel_size=strategies.integers(min_value=1, max_value=10),
)
def test_convolutional_extract_weights(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
) -> None:
    layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )
    weights = extract_layer(layer).weights.weights
    assert weights.shape == (out_channels, in_channels * kernel_size**2 + 1)


@given(
    input_size=strategies.integers(min_value=1, max_value=10),
    hidden_size=strategies.integers(min_value=1, max_value=10),
    number_of_layers=strategies.integers(min_value=1, max_value=10),
)
def test_rnn_extract_weights(
    input_size: int,
    hidden_size: int,
    number_of_layers: int,
) -> None:
    layer = nn.RNN(
        input_size,
        hidden_size,
        num_layers=number_of_layers,
        batch_first=True,
    )
    untyped_layers = [layer.weights for layer in extract_rnn_layers(layer)]
    layers = cast(list[rnn.Weights], untyped_layers)
    assert len(layers) == number_of_layers
    assert layers[0].input_to_hidden.weights.shape == (hidden_size, input_size + 1)

    hidden_to_hidden_shape = (hidden_size, hidden_size + 1)
    assert layers[0].hidden_to_hidden.weights.shape == hidden_to_hidden_shape
    for sub_layer in layers[1:]:
        assert sub_layer.input_to_hidden.weights.shape == hidden_to_hidden_shape
        assert sub_layer.hidden_to_hidden.weights.shape == hidden_to_hidden_shape


def test_extract_scale_isomorphism_after_max_pool() -> None:
    layers = iter([nn.Tanh()])
    isomorphism = extract_scale_isomorphism_after_max_pool(layers)
    assert isomorphism is None
