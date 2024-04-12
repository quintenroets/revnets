from dataclasses import dataclass

from revnets.training import Network, Trainer
from revnets.training.targets import Metrics

from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self) -> None:
        dataset = self.pipeline.load_prepared_data()
        trained_network = self.pipeline.create_target_network()
        network = Network[Metrics](trained_network, learning_rate=0)
        dataloaders = (
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader(),
        )
        trainer = Trainer()
        for dataloader in dataloaders:
            trainer.test(network, dataloader)
