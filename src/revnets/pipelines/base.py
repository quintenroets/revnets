from torch.nn import Sequential

from ..utils import NamedClass


class Pipeline(NamedClass):
    def create_trained_model(self) -> Sequential:
        raise NotImplementedError

    def create_initialized_model(self) -> Sequential:
        raise NotImplementedError

    @classmethod
    def get_base_name(cls):
        return Network.__module__
