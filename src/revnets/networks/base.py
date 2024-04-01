from ..utils import NamedClass
from .models import Model


class Network(NamedClass):
    def create_trained_model(self) -> Model:
        raise NotImplementedError

    def create_initialized_model(self) -> Model:
        raise NotImplementedError

    @classmethod
    def get_base_name(cls):
        return Network.__module__
