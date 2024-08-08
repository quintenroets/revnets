from torch.nn import Sequential

from revnets.utils import NamedClass


class Pipeline(NamedClass):
    def create_target_network(self) -> Sequential:
        raise NotImplementedError  # pragma: nocover

    def create_initialized_network(self) -> Sequential:
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_base_name(cls) -> str:
        return Pipeline.__module__
