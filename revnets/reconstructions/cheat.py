from . import empty


class Reconstructor(empty.Reconstructor):
    def train(self):
        state_dict = self.original.state_dict()
        self.reconstruction.load_state_dict(state_dict)
