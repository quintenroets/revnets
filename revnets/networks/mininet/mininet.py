from ...utils import config
from . import mininet_untrained


class Network(mininet_untrained.Network):
    @classmethod
    def get_trained_network(cls):
        model = super().get_trained_network()
        cls.load_trained_weights(model, config.manual_seed)
        return model
