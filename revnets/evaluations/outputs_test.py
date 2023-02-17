import torch

from ..networks.models.metrics import Phase
from . import outputs_val


def evaluate(original: torch.nn.Module, reconstruction: torch.nn.Module, network):
    return outputs_val.evaluate(original, reconstruction, network, Phase.TEST)
