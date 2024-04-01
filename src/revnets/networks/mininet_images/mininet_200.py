from . import mininet_small


class Network(mininet_small.Network):
    @classmethod
    def initialize_model(cls):
        return cls.get_model_module().Model(hidden_size=200)
