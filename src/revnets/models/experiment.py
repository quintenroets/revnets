from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from package_utils.dataclasses.mixins import SerializationMixin

from .path import Path


@dataclass
class Experiment(SerializationMixin):
    reconstruction_technique: list[str] = field(default_factory=lambda: ["cheat"])
    pipeline: list[str] = field(default_factory=lambda: ["mininet", "mininet"])
    seed: int = 0
    target_network_seed: int = 900

    def __post_init__(self) -> None:
        assert self.seed != self.target_network_seed

    @property
    def names(self) -> tuple[str, ...]:
        seeds = f"{self.seed}_{self.target_network_seed}"
        reconstruction = "_".join(self.reconstruction_technique)
        pipeline = "_".join(self.pipeline)
        return reconstruction, pipeline, seeds

    @property
    def title(self) -> str:
        parts = self.generate_title_parts()
        return " ".join(parts).title()

    def generate_title_parts(self) -> Iterator[str]:
        for name in self.reconstruction_technique:
            yield name.replace("_", " ")
        yield from (
            "|",
            *self.pipeline,
            "|",
            str(self.seed),
            "|",
            str(self.target_network_seed),
        )

    @property
    def path(self) -> Path:
        path = Path.results
        for name in self.names:
            path /= name
        return cast(Path, path)

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    def prepare_config(self, config: dict[str, Any]) -> None:
        self.config_path.yaml = config | {"experiment": self.dict()}
