import pytest
from hypothesis import given, strategies
from revnets.standardization.utils import extract_weights
from torch import nn

from tests.evaluations.test_standardize import activations


@given(
    in_features=strategies.integers(min_value=1, max_value=10),
    out_features=strategies.integers(min_value=1, max_value=10),
)
def test_linear_extract_weights(in_features: int, out_features: int) -> None:
    layer = nn.Linear(in_features=in_features, out_features=out_features)
    weights = extract_weights(layer)
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
    weights = extract_weights(layer)
    assert weights.shape == (out_channels, in_channels * kernel_size**2 + 1)


@pytest.mark.parametrize("activation", activations)
def test_activation_extract_weights(activation: nn.Module) -> None:
    with pytest.raises(AttributeError):
        extract_weights(activation)
