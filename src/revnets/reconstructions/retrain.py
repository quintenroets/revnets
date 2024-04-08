from . import base


class Reconstructor(base.Reconstructor):
    def reconstruct_weights(self) -> None:
        self.pipeline.train(self.reconstruction)
