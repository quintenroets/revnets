import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

from .config import config


class Trainer(pl.Trainer):
    def __init__(
        self,
        accelerator="auto",
        strategy=None,
        logger=None,
        max_epochs=None,
        precision=None,
        **kwargs,
    ):
        if logger is None and config.log:
            logger = config.logger
        if strategy is None and "num_nodes" not in kwargs:
            strategy = DDPStrategy(find_unused_parameters=False)

        callbacks = [RichModelSummary(), RichProgressBar()]

        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            strategy=strategy,
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

    def save_log_message(self):
        config.test_results_path.text = self.log_message

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
