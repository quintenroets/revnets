from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.table import Table


@dataclass
class Evaluation:
    weights_MSE: str = None
    weights_MAE: str = None
    weights_max_AE: str = None
    weights_MAE_layers: str = None
    train_outputs_MAE: str = None
    val_outputs_MAE: str = None
    test_outputs_MAE: str = None
    test_acc: str = None
    adversarial_test_acc: str = None
    adversarial_transfer_test_acc: str = None

    def metric_names(self):
        valid_keys = self.dict().keys()
        return [
            self.format_name(field.name)
            for field in fields(self)
            if field.name in valid_keys
        ]

    @classmethod
    def format_name(cls, name: str):
        name = name.replace("_", " ")
        return name[0].upper() + name[1:]

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def values(self):
        yield from self.dict().values()

    @property
    def table(self) -> Table:
        # slow import
        from ..utils.table import Table

        table = Table(show_lines=True)
        table.add_column("Metric", style="cyan", max_width=20, overflow="fold")

        for name in self.metric_names():
            table.add_column(name, style="magenta", max_width=13)
        table.add_row("Value", *self.values())
        return table

    def show(self) -> None:
        self.table.show()