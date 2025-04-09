from typing import TypeVar, cast

import superpathlib
from simple_classproperty import classproperty
from typing_extensions import Self

T = TypeVar("T", bound="Path")


class Path(superpathlib.Path):
    @classmethod
    @classproperty
    def source_root(cls) -> Self:
        return cls(__file__).parent.parent

    @classmethod
    @classproperty
    def assets(cls) -> Self:
        path = cls.script_assets / cls.source_root.name
        return cast("Self", path)

    @classmethod
    @classproperty
    def config(cls) -> Self:
        path = cls.assets / "config" / "config.yaml"
        return cast("Self", path)

    @classmethod
    @classproperty
    def data(cls) -> Self:
        path = cls.assets / "data"
        return cast("Self", path)

    @classmethod
    @classproperty
    def weights(cls) -> Self:
        path = cls.assets / "weights"
        return cast("Self", path)

    @classmethod
    @classproperty
    def results(cls) -> Self:
        path = cls.assets / "results"
        return cast("Self", path)
