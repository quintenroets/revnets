from . import base


class Reconstructor(base.Reconstructor):
    def reconstruct_weights(self) -> None:
        state_dict = self.pipeline.target.state_dict()
        self.reconstruction.load_state_dict(state_dict)
