from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from package_utils.dataclasses.mixins import SerializationMixin

from .path import Path


@dataclass
class Experiment(SerializationMixin):
    reconstruction_technique: list[str] = field(default_factory=lambda: ["empty"])
    network_to_reconstruct: list[str] = field(
        default_factory=lambda: ["mininet", "mininet"]
    )
    seed: int = 0
    target_network_seed: int = 999

    def __post_init__(self) -> None:
        assert self.seed != self.target_network_seed

    @property
    def names(self) -> tuple[str, ...]:
        seeds = f"{self.seed}_{self.target_network_seed}"
        return (
            "_".join(self.reconstruction_technique),
            "_".join(self.network_to_reconstruct),
            seeds,
        )

    @property
    def name(self) -> str:
        return "_".join(self.names)

    @property
    def title(self) -> str:
        parts = (str(part) for part in self.generate_title_parts())
        return " ".join(parts).title()

    def generate_title_parts(self) -> Iterator[str | int]:
        yield from (
            *self.reconstruction_technique,
            "|",
            *self.network_to_reconstruct,
            "|",
            self.seed,
            "|",
            self.target_network_seed,
        )

    @property
    def path(self) -> Path:
        path = Path.results
        for name in self.names:
            path /= name
        return path

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    def prepare_config(self, config: dict[str, Any]) -> None:
        self.config_path.yaml = config | {"experiment": self.dict()}
