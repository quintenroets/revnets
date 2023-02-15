from . import mininet_untrained


class Network(mininet_untrained.Network):
    @classmethod
    def get_trained_network(cls):
        seed = 27
        model = super().get_trained_network()
        cls.load_trained_weights(model, seed)
        return model
