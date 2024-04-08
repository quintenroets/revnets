from dataclasses import dataclass

from revnets.training import Network, Trainer

from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self) -> None:
        dataset = self.pipeline.create_dataset()
        trained_network = self.pipeline.create_trained_network()
        network = Network(trained_network, learning_rate=0)
        dataset.prepare()
        dataset.calibrate(network)
        dataloaders = (
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader(),
        )
        trainer = Trainer()
        for dataloader in dataloaders:
            trainer.test(network, dataloader)
