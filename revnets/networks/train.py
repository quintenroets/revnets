from ..utils.trainer import Trainer


class Network:
    @classmethod
    def train(cls, network):
        data = cls.dataset()
        data.setup("train")
        Trainer().fit(network, data)

    @classmethod
    def dataset(cls):
        raise NotImplementedError
