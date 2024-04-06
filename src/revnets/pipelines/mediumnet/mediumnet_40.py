from . import mediumnet


class Network(mediumnet.Network):
    @classmethod
    def initialize_model(cls):
        return cls.get_model_module().Model(hidden_size1=40, hidden_size2=20)
