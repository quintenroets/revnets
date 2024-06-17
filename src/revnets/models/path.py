from typing import TypeVar, cast

import superpathlib
from simple_classproperty import classproperty

T = TypeVar("T", bound="Path")


class Path(superpathlib.Path):
    @classmethod
    @classproperty
    def source_root(cls: type[T]) -> T:
        return cls(__file__).parent.parent

    @classmethod
    @classproperty
    def assets(cls: type[T]) -> T:
        path = cls.script_assets / cls.source_root.name
        return cast(T, path)

    @classmethod
    @classproperty
    def config(cls: type[T]) -> T:
        path = cls.assets / "config"
        return cast(T, path)

    @classmethod
    @classproperty
    def data(cls: type[T]) -> T:
        path = cls.assets / "data"
        return cast(T, path)

    @classmethod
    @classproperty
    def weights(cls: type[T]) -> T:
        path = cls.assets / "weights"
        return cast(T, path)

    @classmethod
    @classproperty
    def results(cls: type[T]) -> T:
        path = cls.assets / "results"
        return cast(T, path)
