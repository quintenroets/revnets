from abc import ABC
from typing import cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress import rich_progress
from rich import get_console, reconfigure
from rich.console import RenderableType
from rich.progress import Task, TaskID
from rich.text import Text

from revnets.utils.config import config


class MaxRefresh(rich_progress.ProgressColumn, ABC):
    max_refresh: float = config.console_metrics_refresh_interval

    def __call__(self, task: "Task") -> RenderableType:
        current_time = task.get_time()
        if self.max_refresh is not None:
            try:
                timestamp, renderable = self._renderable_cache[task.id]
            except KeyError:
                pass
            else:
                if timestamp + self.max_refresh > current_time:
                    return renderable

        renderable = self.render(task)
        self._renderable_cache[task.id] = (current_time, renderable)
        return renderable


class MetricsTextColumn(MaxRefresh, rich_progress.MetricsTextColumn):
    def render(self, task: "Task") -> Text:
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[
                    self._current_task_id
                ][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        delimiter = "\n"
        chunks = self.generate_chunks()
        text = delimiter.join(chunks)
        return Text(text, justify="left", style=self._style)

    def generate_chunks(self):
        for k, v in self._metrics.items():
            value = f"{v:.3e}" if isinstance(v, float) else v
            yield f"{k}: {value}"


class TextColumn(MaxRefresh, rich_progress.TextColumn):
    pass


class CustomBarColumn(MaxRefresh, rich_progress.CustomBarColumn):
    pass


class ProcessingSpeedColumn(MaxRefresh, rich_progress.ProcessingSpeedColumn):
    pass


class CustomTimeColumn(MaxRefresh, rich_progress.CustomTimeColumn):
    pass


class BatchesProcessedColumn(MaxRefresh, rich_progress.BatchesProcessedColumn):
    pass


class ProgressBar(RichProgressBar):
    def __init__(self, *args, refresh_rate=100, **kwargs):
        super().__init__(*args, refresh_rate=refresh_rate, **kwargs)

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num")
        return metrics

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
            columns = *self.configure_columns(trainer), self._metric_component
            self.progress = rich_progress.CustomProgress(
                *columns,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def configure_columns(self, trainer: "pl.Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._enabled = False

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._enabled = True

    def on_sanity_check_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pass
