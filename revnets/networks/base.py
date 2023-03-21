from ..utils import NamedClass


class Network(NamedClass):
    @classmethod
    def get_trained_network(cls):
        raise NotImplementedError

    @classmethod
    def get_architecture(cls):
        raise NotImplementedError

    @classmethod
    def get_base_name(cls):
        return Network.__module__
