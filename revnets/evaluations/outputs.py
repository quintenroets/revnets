import pytorch_lightning as pl
import torch
import torchmetrics
from cacher.caches.speedup_deep_learning import cache
from rich.pretty import pprint

from ..data.mnist1d import Dataset
from ..utils.trainer import Trainer


class CompareModel(pl.LightningModule):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.mse = None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        models = (self.model1, self.model2)
        outputs = (model(inputs) for model in models)
        self.mse_metric.update(*outputs)

    def on_test_epoch_end(self) -> None:
        self.mse = self.mse_metric.compute().item()


@cache
def compare_outputs(model1, model2, dataset):
    dataset.setup("valid")
    model = CompareModel(model1, model2)
    dataset.calibrate(model)
    dataloader = dataset.val_dataloader()
    Trainer().test(model, dataloaders=dataloader)
    return model.mse


def evaluate(original: torch.nn.Module, reconstruction: torch.nn.Module, network):
    dataset: Dataset = network.dataset()
    difference = compare_outputs(original, reconstruction, dataset)
    message = f"Output difference: {difference}"
    pprint(message)
