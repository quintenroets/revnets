from __future__ import annotations

import typing
from typing import TypeVar

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
        return typing.cast(T, path)

    @classmethod
    @classproperty
    def config(cls: type[T]) -> T:
        path = cls.assets / "config"
        return typing.cast(T, path)

    @classmethod
    @classproperty
    def data(cls: type[T]) -> T:
        path = cls.assets / "data"
        return typing.cast(T, path)

    @classmethod
    @classproperty
    def weights(cls: type[T]) -> T:
        path = cls.assets / "weights"
        return typing.cast(T, path)

    @classmethod
    @classproperty
    def results(cls: type[T]) -> T:
        path = cls.assets / "results"
        return typing.cast(T, path)
