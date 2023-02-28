import torch

from .standardize import standardize


def evaluate(original: torch.nn.Module, reconstruction: torch.nn.Module, *_, **__):
    return get_mse(original, reconstruction)


def same_architecture(original: torch.nn.Module, reconstruction: torch.nn.Module):
    return all(
        original_weights.shape == reconstructed_weights.shape
        for original_weights, reconstructed_weights in zip(
            original.state_dict().values(), reconstruction.state_dict().values()
        )
    )


def get_mse(original: torch.nn.Module, reconstruction: torch.nn.Module):
    if not same_architecture(original, reconstruction):
        mse = None
    else:
        standardize.standardize(original)
        standardize.standardize(reconstruction)
        mse = calculate_mse(original, reconstruction)
    return mse


def calculate_mse(original, reconstruction):
    MSE_size = sum(weights.numel() for weights in original.state_dict().values())
    MSE_sum = sum(
        torch.nn.functional.mse_loss(
            original_weights, reconstructed_weights, reduction="sum"
        ).item()
        for original_weights, reconstructed_weights in zip(
            original.state_dict().values(), reconstruction.state_dict().values()
        )
    )
    mse = MSE_sum / MSE_size
    return mse
