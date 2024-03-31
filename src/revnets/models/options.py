from dataclasses import dataclass

from .path import Path


@dataclass
class Options:
    debug: bool = True
    config_path: Path = Path.config
