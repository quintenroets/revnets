from .. import models
from ..mininet import mininet


class Network(mininet.Network):
    @classmethod
    def get_model_module(cls):
        return models.mediumnet
