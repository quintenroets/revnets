from . import base


class Reconstructor(base.Reconstructor):
    def reconstruct_weights(self):
        state_dict = self.original.state_dict()
        self.reconstruction.load_state_dict(state_dict)
