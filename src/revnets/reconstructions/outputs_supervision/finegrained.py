from dataclasses import dataclass

from revnets.data import output_supervision

from .iterative_sampling import difficult_train_inputs


@dataclass
class Reconstructor(difficult_train_inputs.Reconstructor):
    n_rounds: int = 2

    def train_model(self, data: output_supervision.Dataset) -> None:
        # self.model.use_loss_sum = True
        # self.model.learning_rate = 10 ** -9
        super().train_model(data)
