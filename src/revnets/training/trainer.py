from collections.abc import Iterator

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, RichModelSummary
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)

from ..context import context


class Trainer(pl.Trainer):
    def __init__(
        self,
        accelerator: str = "auto",
        callbacks: list[Callback] = None,
        logger=None,
        max_epochs=None,
        precision=None,
        barebones: bool = False,
        **kwargs,
    ) -> None:
        config = context.config
        callbacks = list(self.generate_callbacks(callbacks, barebones))
        super().__init__(
            accelerator=accelerator,
            barebones=barebones,
            callbacks=callbacks,
            logger=logger,
            max_epochs=max_epochs or context.config.reconstruction_training.epochs,
            num_sanity_val_steps=config.number_of_validation_sanity_steps,
            limit_train_batches=config.limit_batches,
            limit_val_batches=config.limit_batches,
            limit_test_batches=config.limit_batches,
            default_root_dir=context.log_path_str,
            precision=precision or config.precision,
            sync_batchnorm=True,
            # gradient_clip_val=config.gradient_clip_val,
            **kwargs,
        )

    @classmethod
    def generate_callbacks(
        cls, callbacks: list[Callback], barebones: bool
    ) -> Iterator[Callback]:
        if callbacks is not None:
            yield from callbacks
        extra_callbacks = not barebones and not context.is_running_in_notebook
        if extra_callbacks:
            yield RichModelSummary()
            theme = RichProgressBarTheme(
                metrics_format=".3e", metrics_text_delimiter="\n"
            )
            yield RichProgressBar(theme=theme)

    @property
    def log_message(self) -> str:
        # slow import
        from ..utils.table import Table  # noqa: autoimport

        table = Table()
        table.add_column("Test metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in self.logged_metrics.items():
            if name != "step":
                name = name.replace("test ", "")
                value = value.item()
                display_format = ".3%" if "accuracy" in name else ".3f"
                value = f"{value:{display_format}}".replace("%", " %")
                table.add_row(name, value)

        return table.text
