import torch.nn

from . import mininet


class Network(mininet.Network):
    def train(self, model: torch.nn.Module) -> None:
        pass
