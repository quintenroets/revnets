from . import base


class Reconstructor(base.Reconstructor):
    def reconstruct_weights(self):
        self.network.load_trained_weights(self.reconstruction)
