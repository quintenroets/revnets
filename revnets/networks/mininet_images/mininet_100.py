from revnets import data

from .. import models
from ..mininet import mininet


class Network(mininet.Network):
    @classmethod
    def get_model_module(cls):
        return models.mininet_images

    @classmethod
    def initialize_model(cls):
        return cls.get_model_module().Model(hidden_size=40)

    @classmethod
    def dataset(cls):
        return data.mnist.Dataset()
