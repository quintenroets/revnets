class NamedClass:
    @classmethod
    def get_base_name(cls):
        raise NotImplementedError

    @classmethod
    @property
    def name(cls):  # noqa
        base_name = cls.get_base_name()
        last_dot = base_name.rfind(".")
        base_name = base_name[: last_dot + 1]
        name = cls.__module__.replace(base_name, "")
        for token in "_/.":
            name = name.replace(token, " ")
        return name.capitalize()
