from simple_classproperty import classproperty

from ...data import mnist1d
from .. import train
from ..models import mininet


class Network(train.Network):
    @classmethod
    def create_model(cls):
        return cls.model_module.Model()

    @classmethod
    @classproperty
    def model_module(cls):
        return mininet

    @classmethod
    def create_dataset(cls):
        return mnist1d.Dataset()
