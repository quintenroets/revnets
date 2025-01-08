from collections.abc import Iterator
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, RichModelSummary
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)

from revnets.context import context


class Trainer(pl.Trainer):
    def __init__(
        self,
        accelerator: str = "auto",
        callbacks: list[Callback] | None = None,
        max_epochs: int | None = None,
        precision: int | None = None,
        *,
        barebones: bool = False,
        **kwargs: Any,
    ) -> None:
        config = context.config
        callbacks = list(self.generate_callbacks(callbacks, barebones=barebones))
        super().__init__(
            accelerator=accelerator,
            barebones=barebones,
            callbacks=callbacks,
            max_epochs=max_epochs or context.config.reconstruction_training.epochs,
            num_sanity_val_steps=config.number_of_validation_sanity_steps,
            limit_train_batches=config.limit_batches,
            limit_val_batches=config.limit_batches,
            limit_test_batches=config.limit_batches,
            default_root_dir=context.log_path_str,
            precision=precision or config.precision,  # type: ignore[arg-type]
            sync_batchnorm=True,
            gradient_clip_val=context.config.gradient_clip_val,
            **kwargs,
        )

    @classmethod
    def generate_callbacks(
        cls,
        callbacks: list[Callback] | None,
        *,
        barebones: bool,
    ) -> Iterator[Callback]:
        if callbacks is not None:
            yield from callbacks
        extra_callbacks = not barebones and not context.is_running_in_notebook
        if extra_callbacks:
            yield RichModelSummary()
            theme = RichProgressBarTheme(
                metrics_format=".3e",
                metrics_text_delimiter="\n",
            )
            yield RichProgressBar(theme=theme)
