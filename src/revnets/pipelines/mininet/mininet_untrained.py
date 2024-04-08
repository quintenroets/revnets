import torch.nn

from . import mininet


class Pipeline(mininet.Pipeline):
    def train(self, model: torch.nn.Module) -> None:
        pass
