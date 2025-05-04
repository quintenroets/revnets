from dataclasses import dataclass

from revnets.models import Path


@dataclass
class Options:
    config_path: Path = Path.config / "config.yaml"
