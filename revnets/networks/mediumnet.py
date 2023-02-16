from . import mininet
from .models import mediumnet as mediumnet_model


class Network(mininet.Network):
    @classmethod
    def get_model_module(cls):
        return mediumnet_model
