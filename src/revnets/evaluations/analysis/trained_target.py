from dataclasses import dataclass

from revnets.training import Trainer
from revnets.training.targets import Network

from .. import base


@dataclass
class Evaluator(base.Evaluator):
    def evaluate(self) -> None:
        dataset = self.pipeline.load_prepared_data()
        network = Network(self.pipeline.target)
        dataloaders = (
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader(),
        )
        trainer = Trainer()
        for dataloader in dataloaders:
            trainer.test(network, dataloader)
