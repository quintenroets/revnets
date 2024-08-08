from types import ModuleType

import pytest
from pytorch_lightning.core.optimizer import LightningOptimizer
from revnets import reconstructions
from revnets.pipelines.mininet import Pipeline
from revnets.reconstructions import Reconstructor, queries
from revnets.training.reconstructions import Network
from revnets.training.reconstructions.callbacks import LearningRateScheduler
from torch import nn

reconstruction_modules = (
    reconstructions.empty,
    reconstructions.cheat,
    reconstructions.retrain,
    queries.random,
    queries.correlated_features,
    queries.arbitrary_correlated_features,
    queries.target_train_data,
    queries.iterative.difficult_inputs,
    queries.iterative.difficult_train_inputs,
)


@pytest.mark.parametrize("reconstruction_module", reconstruction_modules)
@pytest.mark.usefixtures("test_context")
def test_reconstructions(reconstruction_module: ModuleType) -> None:
    pipeline = Pipeline()
    reconstructor: Reconstructor = reconstruction_module.Reconstructor(pipeline)
    reconstructor.create_reconstruction()


def test_learning_rate_scheduler() -> None:
    model = nn.Linear(in_features=1, out_features=1)
    network = Network(model)
    LearningRateScheduler.check_learning_rate(network, 0)
    optimizer = network.configure_optimizers()
    lightning_optimizer = LightningOptimizer(optimizer)
    network.update_optimizer(lightning_optimizer, 0)
