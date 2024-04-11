from dataclasses import dataclass

from revnets.training import Network, Trainer

from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self) -> None:
        dataset = self.pipeline.load_prepared_data()
        trained_network = self.pipeline.create_trained_network()
        network = Network(trained_network, learning_rate=0)
        dataloaders = (
            dataset.train_dataloader(),
            dataset.val_dataloader(),
            dataset.test_dataloader(),
        )
        trainer = Trainer()
        for dataloader in dataloaders:
            trainer.test(network, dataloader)
