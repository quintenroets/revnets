from . import mininet


class Network(mininet.Network):
    @classmethod
    def initialize_model(cls):
        return cls.get_model_module().Model(hidden_size=100)
