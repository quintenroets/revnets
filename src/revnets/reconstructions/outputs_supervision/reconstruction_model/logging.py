from typing import Any, cast

from revnets.evaluations import weights

from ....evaluations.weights.mae import Evaluator
from . import network


class ReconstructNetwork(network.ReconstructNetwork):
    def __init__(
        self,
        *args: Any,
        visualize: bool = False,
        visualization_interval: int = 10,
        calculate_interval: int = 5,
    ) -> None:
        super().__init__(*args)
        self.visualization_interval = visualization_interval
        self.visualizer = weights.visualizer.Evaluator(self.model, self.pipeline)
        self.do_visualize = visualize
        self.calculate_interval = calculate_interval
        self.visualize(before_training=True)
        self.mae = self.get_weights_MAE()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.calculate_interval == 0:
            self.calculate_weights_MAE()
            self.log("weights MAE", self.mae, prog_bar=True)

    def calculate_weights_MAE(self) -> None:
        state_dict = self.model.state_dict()
        self.mae = self.get_weights_MAE()
        # undo standardize operations on state dict
        self.model.load_state_dict(state_dict)

    def get_weights_MAE(self) -> float:
        evaluator = Evaluator(self.model, self.pipeline)
        result = evaluator.evaluate()
        return cast(float, result)

    def visualize(self, before_training: bool = False) -> None:
        if self.do_visualize:
            should_visualize = before_training or self.is_visualization_epoch
            if should_visualize:
                self.visualizer.evaluate()

    @property
    def is_visualization_epoch(self) -> bool:
        return (self.trainer.current_epoch + 1) % self.visualization_interval == 0
