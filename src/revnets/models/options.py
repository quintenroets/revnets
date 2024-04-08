from dataclasses import dataclass

from .path import Path


@dataclass
class Options:
    config_path: Path = Path.config
    experiment: str | None = None
