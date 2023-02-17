from ..networks.models import mininet
from . import retrain_on_outputs


class Reconstructor(retrain_on_outputs.Reconstructor):
    def initialize_reconstruction(self):
        return mininet.Model(hidden_size=256)
