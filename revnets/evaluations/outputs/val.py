import pytorch_lightning as pl
import torchmetrics
from cacher.caches.speedup_deep_learning import cache

from ...utils.trainer import Trainer
from .. import base


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


class Evaluator(base.Evaluator):
    def evaluate(self):
        return self.compare_outputs()

    @cache
    def compare_outputs(self):
        model = CompareModel(self.original, self.reconstruction)

        dataset = self.get_dataset()
        dataset.setup("valid")
        dataset.calibrate(model)

        dataloader = self.get_dataloader(dataset)

        Trainer().test(model, dataloaders=dataloader)  # noqa
        return model.mse

    @classmethod
    def get_dataloader(cls, dataset):
        return dataset.val_dataloader()
