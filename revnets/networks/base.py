from ..utils import Path


class Network:
    @classmethod
    def get_trained_network(cls):
        raise NotImplementedError

    @classmethod
    def get_architecture(cls):
        raise NotImplementedError

    @classmethod
    @property
    def name(cls):  # noqa
        filename = Path(__file__).stem
        base_name = Network.__module__.replace(filename, "")
        name = cls.__module__.replace(base_name, "")
        for token in "_/.":
            name = name.replace(token, " ")
        return name.capitalize()
