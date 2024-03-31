from dataclasses import dataclass, field
from typing import Annotated

from typer import Option

from .path import Path

devices_help = "Indices of GPU devices to use"


@dataclass
class Options:
    config_path: Path = Path.config
    config_name: str | None = None
    experiment_name: str | None = None
    seed: str | None = None
    seed_range: list[int] = field(default_factory=list)
    gpu_devices: Annotated[list[int], Option(help=devices_help)] = field(
        default_factory=list
    )
