import contextlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar

from . import pl_logger
from .config import config


class Trainer(pl.Trainer):
    def __init__(
        self,
        accelerator="auto",
        callbacks=None,
        logger=None,
        max_epochs=None,
        precision=None,
        barebones=False,
        **kwargs,
    ):
        if config.quiet_prediction:
            pl_logger.Quiet().__enter__()  # remove logging messages
        if logger is None and config.log:
            logger = config.logger
        try:
            get_ipython()
            is_notebook = True
        except NameError:
            is_notebook = False

        extra_callbacks = (
            [] if barebones or is_notebook else [RichModelSummary(), RichProgressBar()]
        )
        callbacks = (callbacks or []) + extra_callbacks

        super().__init__(
            accelerator=accelerator,
            barebones=barebones,
            callbacks=callbacks,
            logger=logger,
            max_epochs=max_epochs or config.epochs,
            num_sanity_val_steps=config.num_sanity_val_steps,
            limit_train_batches=config.limit_batches,
            limit_val_batches=config.limit_batches,
            limit_test_batches=config.limit_batches,
            default_root_dir=str(config.log_folder),
            precision=precision or config.precision,
            sync_batchnorm=True,
            **kwargs,
        )

    @property
    def context_manager(self):
        return (
            pl_logger.Quiet() if config.quiet_prediction else contextlib.nullcontext()
        )

    def predict(self, *args, **kwargs):
        with self.context_manager:
            return super().predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        with self.context_manager:
            return super().fit(*args, **kwargs)

    @property
    def log_message(self):
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
