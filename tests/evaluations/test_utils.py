import pytest
from hypothesis import given, strategies
from revnets.standardization.utils import extract_parameters, extract_weights
from torch import nn

from tests.evaluations.test_standardize import activations


@given(
    in_features=strategies.integers(min_value=1, max_value=10),
    out_features=strategies.integers(min_value=1, max_value=10),
)
def test_linear_extract_weights(in_features: int, out_features: int) -> None:
    layer = nn.Linear(in_features=in_features, out_features=out_features)
    parameters = extract_parameters(layer)
    weights = extract_weights(parameters)
    assert weights.shape == (out_features, in_features + 1)


@given(
    in_channels=strategies.integers(min_value=1, max_value=10),
    out_channels=strategies.integers(min_value=1, max_value=10),
    kernel_size=strategies.integers(min_value=1, max_value=10),
)
def test_convolutional_extract_weights(
    in_channels: int, out_channels: int, kernel_size: int
) -> None:
    layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
    )
    parameters = extract_parameters(layer)
    weights = extract_weights(parameters)
    assert weights.shape == (out_channels, in_channels * kernel_size**2 + 1)


"""
@given(
    input_size=strategies.integers(min_value=1, max_value=10),
    hidden_size=strategies.integers(min_value=1, max_value=10),
)
"""


# def test_rnn_extract_weights(input_size: int, hidden_size: int) -> None:
def test_rnn_extract_weights() -> None:
    layer = nn.RNN(16, 25, num_layers=3)
    for name, p in layer.named_parameters():
        pprint(name)
        pprint(p.shape)
    parameters = extract_parameters(layer)
    weights = extract_weights(parameters)
    pprint(weights.shape)
    # assert weights.shape == (


@pytest.mark.parametrize("activation", activations)
def test_activation_extract_weights(activation: nn.Module) -> None:
    with pytest.raises(AttributeError):
        extract_parameters(activation)
