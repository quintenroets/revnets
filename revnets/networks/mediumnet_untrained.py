from . import mininet_untrained
from .models import mediumnet as mediumnet_model


class Network(mininet_untrained.Network):
    @classmethod
    def get_model_module(cls):
        return mediumnet_model
