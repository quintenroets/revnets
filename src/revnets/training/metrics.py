from dataclasses import dataclass, fields

import torch
from simple_classproperty import classproperty


@dataclass
class Metrics:
    loss: torch.Tensor

    def dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.names}

    @classmethod
    @classproperty
    def names(cls) -> list[str]:
        return [field.name for field in fields(cls)]
