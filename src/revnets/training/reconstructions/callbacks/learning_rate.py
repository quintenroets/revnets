from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from pytorch_lightning import Callback, LightningModule, Trainer

from revnets.context import context

if TYPE_CHECKING:
    from revnets.training.reconstructions.network import Network

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


@dataclass
class LearningRateScheduler(Callback):
    losses: list[float] = field(default_factory=list)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.log(
            "learning rate",
            pl_module.learning_rate,
            prog_bar=True,
            on_step=False,
        )
        loss = trainer.callback_metrics["train l1 loss"].item()
        self.losses.append(loss)
        self.check_learning_rate(pl_module, loss)

    @classmethod
    def check_learning_rate(cls, pl_module: LightningModule, loss: float) -> None:
        target_scale = 1.0
        for threshold, scale in transitions.items():
            if loss < threshold:
                target_scale = scale
        initial_learning_rate = context.config.target_network_training.learning_rate
        learning_rate = initial_learning_rate * target_scale
        if learning_rate < pl_module.learning_rate:
            network = cast("Network", pl_module)
            network.learning_rate = learning_rate
