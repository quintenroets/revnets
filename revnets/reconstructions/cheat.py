import torch.nn


def reconstruct(original: torch.nn.Module, reconstructed: torch.nn.Module):
    reconstructed.load_state_dict(original.state_dict())
