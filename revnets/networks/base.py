from functools import cached_property

from ..utils import NamedClass


class Network(NamedClass):
    @cached_property
    def trained_network(self):
        return self.get_trained_network()

    @classmethod
    def get_trained_network(cls):
        raise NotImplementedError

    @classmethod
    def get_architecture(cls, seed=None):
        raise NotImplementedError

    @classmethod
    def get_base_name(cls):
        return Network.__module__
