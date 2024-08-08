from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import cli

if TYPE_CHECKING:  # pragma: nocover
    from collections.abc import Iterator

    from rich.table import Table


@dataclass
class Evaluation:
    weights_MSE: str | None = None  # noqa: N815
    weights_MAE: str | None = None  # noqa: N815
    weights_max_AE: str | None = None  # noqa: N815
    weights_MAE_layers: str | None = None  # noqa: N815
    train_outputs_MAE: str | None = None  # noqa: N815
    val_outputs_MAE: str | None = None  # noqa: N815
    test_outputs_MAE: str | None = None  # noqa: N815
    test_acc: str | None = None
    adversarial_test_acc: str | None = None
    adversarial_transfer_test_acc: str | None = None

    def metric_names(self) -> list[str]:
        valid_keys = self.dict().keys()
        return [
            self.format_name(field.name)
            for field in fields(self)
            if field.name in valid_keys
        ]

    @classmethod
    def format_name(cls, name: str) -> str:
        name = name.replace("_", " ")
        return name[0].upper() + name[1:]

    def dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def values(self) -> Iterator[str]:
        yield from self.dict().values()

    def create_table(self) -> Table:
        # slow import
        from rich.table import Table

        table = Table(show_lines=True)
        table.add_column("Metric", style="cyan", max_width=20, overflow="fold")

        for name in self.metric_names():
            table.add_column(name, style="magenta", max_width=13)
        table.add_row("Value", *self.values())
        return table

    def show(self) -> None:
        table = self.create_table()
        cli.console.print(table)
