from . import empty


class Reconstructor(empty.Reconstructor):
    def train(self):
        self.network.load_trained_weights(self.reconstruction)
