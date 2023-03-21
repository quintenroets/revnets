import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch import nn

from ...networks.models.metrics import Metrics
from ...utils import config


class LossMetric(torchmetrics.Metric):
    def __iter__(self):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, labels):
        self.loss_sum += nn.functional.cross_entropy(outputs, labels, reduction="sum")
        self.num_examples += len(outputs)

    def compute(self):
        return self.loss_sum / self.num_examples


class RunningMetrics:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)
        self.loss = LossMetric().to(device)

    def compute(self):
        return Metrics(
            accuracy=self.accuracy.compute().item(), loss=self.loss.compute().item()
        )


class AttackModel(pl.LightningModule):
    def __init__(self, original, reconstruction):
        super().__init__()
        self.original = original
        self.reconstruction = reconstruction
        self.model_under_attack = None

        self.test_metric = RunningMetrics()
        self.adversarial_metric = RunningMetrics()
        self.adversarial_transfer_metric = RunningMetrics()

        self.test, self.adversarial, self.adversarial_transfer = (None,) * 3
        self.attack = None
        self.visualize = config.visualize_attack

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        adversarial_inputs = self.get_adversarial_inputs(inputs)

        if self.visualize:
            self.show_comparison(inputs, adversarial_inputs)

        self.evaluate_inputs(self.reconstruction, inputs, labels, self.test_metric)
        self.evaluate_inputs(
            self.reconstruction, adversarial_inputs, labels, self.adversarial_metric
        )
        self.evaluate_inputs(
            self.original, adversarial_inputs, labels, self.adversarial_transfer_metric
        )

    @classmethod
    def show_comparison(cls, inputs, adversarial_inputs):
        length = len(inputs[0])
        indices = np.flip(np.arange(length))

        for image, adversarial in zip(inputs, adversarial_inputs):
            plt.plot(image, indices, color="green", label="original")
            plt.plot(adversarial, indices, color="red", label="adversarial")
            plt.legend()
            plt.show()

    def get_adversarial_inputs(self, inputs):
        if self.model_under_attack is None:
            self.configure_attack(inputs)

        attack_inputs = inputs.cpu().numpy()
        with torch.inference_mode(mode=False):
            adversarial_inputs = self.attack.generate(x=attack_inputs)
        return torch.Tensor(adversarial_inputs).to(inputs.device)

    @classmethod
    def evaluate_inputs(cls, model, inputs, labels, metric: RunningMetrics):
        outputs = model(inputs)
        _, predictions = outputs.max(1)
        metric.accuracy.update(predictions, labels)
        metric.loss.update(outputs, labels)

    def configure_attack(self, inputs):
        outputs = self.reconstruction(inputs)
        self.model_under_attack = PyTorchClassifier(
            model=self.reconstruction,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=self.reconstruction.configure_optimizers(),
            input_shape=inputs.shape[1:],
            nb_classes=outputs.shape[-1],
            device_type="gpu",
        )
        self.attack = FastGradientMethod(
            self.model_under_attack, eps=config.adversarial_epsilon
        )

    def on_test_epoch_end(self) -> None:
        self.test = self.test_metric.compute()
        self.adversarial = self.adversarial_metric.compute()
        self.adversarial_transfer = self.adversarial_transfer_metric.compute()
