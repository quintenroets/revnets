from __future__ import annotations

import os
import typing

from pytorch_lightning import loggers

from .rank import rank_zero_only

if typing.TYPE_CHECKING:
    from ..utils import Config  # noqa: autoimport


@rank_zero_only
def get_logger(config: Config):
    if config.debug and not config.log_on_debug:
        logger = None
    else:
        os.environ["WANDB_SILENT"] = "true"
        project_name = "lightning_logs_debug" if config.debug else "lightning_logs"
        tensorboard_logger = loggers.TensorBoardLogger(
            name=project_name, save_dir=config.save_dir
        )
        wandb_logger = loggers.WandbLogger(
            save_dir=config.save_dir,
            name=config.base_name + f"_version{tensorboard_logger.version}",
            project=project_name,
        )
        logger = (tensorboard_logger, wandb_logger)
    return logger
