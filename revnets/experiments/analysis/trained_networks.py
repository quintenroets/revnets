from dataclasses import dataclass

from revnets.utils.trainer import Trainer

from ...networks.models import trainable
from ...networks.train import Network
from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    network: Network = None

    def run_network(self):
        dataset = self.network.dataset()
        model = trainable.Model(self.network.trained_network)
        dataset.prepare()
        dataset.calibrate(model)
        trainer = Trainer()
        dataloaders = (
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader(),
        )
        for dataloader in dataloaders:
            trainer.test(model, dataloader)
