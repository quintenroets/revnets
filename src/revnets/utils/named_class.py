from simple_classproperty import classproperty


class NamedClass:
    @classmethod
    def get_base_name(cls):
        raise NotImplementedError

    @classmethod
    @classproperty
    def name(cls) -> str:
        return " ".join(cls.relative_module).capitalize()

    @classmethod
    @classproperty
    def relative_module(cls) -> list[str]:
        base_name = cls.extract_base_name()
        return cls.__module__.replace(base_name, "").split(".")

    @classmethod
    def extract_base_name(cls) -> str:
        name = cls.get_base_name()
        last_dot = name.rfind(".")
        return name[: last_dot + 1]
