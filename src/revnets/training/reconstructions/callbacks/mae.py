from dataclasses import dataclass, field
from typing import cast

from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn

from revnets.evaluations import weights
from revnets.evaluations.weights import Evaluator
from revnets.pipelines import Pipeline
from revnets.standardization.network import Standardizer


@dataclass
class MAECalculator(Callback):
    reconstruction: nn.Module
    pipeline: Pipeline
    name: str = "weights MAE"
    calculate_interval: int = 20
    evaluator: Evaluator = field(init=False)

    def __post_init__(self) -> None:
        self.evaluator = weights.mae.Evaluator(self.reconstruction, self.pipeline)
        Standardizer(self.pipeline.target).run()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.calculate_interval == 0:
            mae = self.calculate_mae()
            pl_module.log(self.name, mae, prog_bar=True, on_step=False)

    def calculate_mae(self) -> float:
        state_dict = self.reconstruction.state_dict()
        Standardizer(self.reconstruction).run()
        mae = self.evaluator.calculate_total_distance()
        self.reconstruction.load_state_dict(state_dict)
        return cast("float", mae)
