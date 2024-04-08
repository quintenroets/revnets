from typing import Any

from revnets.context import context

from . import stabilize


class ReconstructNetwork(stabilize.ReconstructNetwork):
    def __init__(
        self, *args: Any, schedule_learning_rate: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.schedule_learning_rate = schedule_learning_rate
        self.losses: list[float] = []

    def log_learning_rate(self) -> None:
        self.log("learning rate", self.learning_rate, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            self.log_learning_rate()
        if self.schedule_learning_rate:
            self.losses.append(self.logged_loss)
            self.check_learning_rate()
        super().on_train_epoch_end()

    @property
    def logged_loss(self) -> float:
        metrics = self.trainer.callback_metrics
        return metrics["train l1 loss"].item()

    def check_learning_rate(self) -> None:
        transitions = {
            1: 1e-1,
            5e-1: 5e-2,
            3e-1: 3e-2,
            2e-1: 2e-2,
            1e-1: 1e-2,
            2e-2: 2e-3,
            1e-2: 1e-3,
            2e-3: 2e-4,
            1e-3: 1e-4,
            2e-4: 2e-5,
            1e-4: 1e-5,
            2e-5: 2e-6,
            1e-5: 1e-7,
            1e-6: 1e-8,
            1e-7: 1e-9,
            1e-9: 1e-10,
            1e-10: 3e-11,
            3e-11: 1e-11,
            1e-11: 3e-12,
        }

        target_scale = 1.0
        for threshold, scale in transitions.items():
            if self.logged_loss < threshold:
                target_scale = scale

        target_learning_rate = (
            context.config.target_network_training.learning_rate * target_scale
        )
        if target_learning_rate < self.learning_rate:
            self.learning_rate = target_learning_rate
            self.log_learning_rate()
