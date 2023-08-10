from revnets.evaluations import weights

from ....evaluations.weights.mae import Evaluator
from . import model


class ReconstructModel(model.ReconstructModel):
    def __init__(
        self,
        *args,
        visualize=False,
        visualization_interval: int = 10,
        calculate_interval: int = 5,
    ):
        super().__init__(*args)
        self.visualization_interval = visualization_interval
        self.visualizer = weights.visualizer.Evaluator(self.model, self.network)
        self.do_visualize = visualize
        self.calculate_interval = calculate_interval
        self.visualize(before_training=True)
        self.mae = self.get_weights_MAE()

    def on_train_epoch_end(self):
        if self.current_epoch % self.calculate_interval == 0:
            self.calculate_weights_MAE()
            self.log("weights MAE", self.mae, prog_bar=True)

    def calculate_weights_MAE(self) -> None:
        state_dict = self.model.state_dict()
        self.mae = self.get_weights_MAE()
        # undo standardize operations on state dict
        self.model.load_state_dict(state_dict)

    def get_weights_MAE(self) -> float:
        evaluator = Evaluator(self.model, self.network)
        return evaluator.evaluate()

    def visualize(self, before_training=False):
        if self.do_visualize:
            should_visualize = before_training or self.is_visualization_epoch
            if should_visualize:
                self.visualizer.evaluate()

    @property
    def is_visualization_epoch(self):
        return (self.trainer.current_epoch + 1) % self.visualization_interval == 0
