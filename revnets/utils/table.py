import cli
from rich import table

from .path import Path


class Table(table.Table):
    @property
    def text(self):
        from rich.console import Console  # noqa: autoimport

        with Path.tempfile() as tmp:
            with tmp.open("w") as fp:
                console = Console(record=True, file=fp, force_terminal=True)
                console.print(self)
                message = console.export_text(styles=True)
                return message

    def show(self):
        cli.console.print(self, highlight=False)
