from . import base


class Reconstructor(base.Reconstructor):
    def reconstruct_weights(self) -> None:
        state_dict = self.pipeline.create_trained_network().state_dict()
        self.reconstruction.load_state_dict(state_dict)
