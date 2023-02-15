import torch.nn


def reconstruct(original: torch.nn.Module, reconstructed: torch.nn.Module, network):
    network.load_trained_weights(reconstructed)
