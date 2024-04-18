from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from numpy.typing import NDArray
from torch import nn
from torch.nn import Module

from revnets.context import context
from revnets.training.targets import Metrics


class LossMetric(torchmetrics.Metric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss_sum += nn.functional.cross_entropy(outputs, labels, reduction="sum")
        self.num_examples += len(outputs)

    def compute(self) -> torch.Tensor:
        value = self.loss_sum / self.num_examples
        return cast(torch.Tensor, value)


class RunningMetrics:
    def __init__(self) -> None:
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=10).to(
            context.device
        )
        self.loss = LossMetric().to(context.device)

    def compute(self) -> Metrics:
        accuracy = self.accuracy.compute().item()  # type: ignore[func-returns-value]
        loss = self.loss.compute()
        return Metrics(loss, accuracy)


class AttackNetwork(pl.LightningModule):
    def __init__(self, original: Module, reconstruction: Module) -> None:
        super().__init__()
        self.original = original
        self.reconstruction = reconstruction
        self.model_under_attack = None

        self.test_metric = RunningMetrics()
        self.adversarial_metric = RunningMetrics()
        self.adversarial_transfer_metric = RunningMetrics()

        self.test: Metrics | None = None
        self.adversarial: Metrics | None = None
        self.adversarial_transfer: Metrics | None = None
        self.attack: FastGradientMethod | None = None
        self.visualize = context.config.evaluation.visualize_attack

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
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
    def show_comparison(
        cls, inputs: torch.Tensor, adversarial_inputs: torch.Tensor
    ) -> None:
        inputs_numpy = cls.extract_visualization_values(inputs)
        adversarial_inputs_numpy = cls.extract_visualization_values(adversarial_inputs)
        length = inputs_numpy.shape[1]
        indices = np.flip(np.arange(length))

        for image, adversarial in zip(inputs_numpy, adversarial_inputs_numpy):
            plt.plot(image, indices, color="green", label="original")
            plt.plot(adversarial, indices, color="red", label="adversarial")
            plt.legend()
            plt.show()

    @classmethod
    def extract_visualization_values(
        cls, values: torch.Tensor, max_elements: int = 10
    ) -> NDArray[np.float64]:
        values_numpy = values.detach().cpu().numpy()[:max_elements]
        values_numpy = values_numpy.reshape(values_numpy.shape[0], -1)
        return cast(NDArray[np.float64], values_numpy)

    def get_adversarial_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model_under_attack is None:
            self.configure_attack(inputs)

        attack_inputs = inputs.cpu().numpy()
        assert self.attack is not None
        with torch.inference_mode(mode=False):
            adversarial_inputs = self.attack.generate(x=attack_inputs)
        return torch.Tensor(adversarial_inputs).to(inputs.device)

    @classmethod
    def evaluate_inputs(
        cls,
        model: Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        metric: RunningMetrics,
    ) -> None:
        outputs = model(inputs)
        _, predictions = outputs.max(1)
        metric.accuracy.update(predictions, labels)
        metric.loss.update(outputs, labels)

    def configure_attack(self, inputs: torch.Tensor) -> None:
        outputs = self.reconstruction(inputs)
        self.model_under_attack = PyTorchClassifier(
            model=self.reconstruction,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=inputs.shape[1:],
            nb_classes=outputs.shape[-1],
            device_type="gpu",
        )
        self.attack = FastGradientMethod(
            self.model_under_attack, eps=context.config.evaluation.adversarial_epsilon
        )

    def on_test_epoch_end(self) -> None:
        self.test = self.test_metric.compute()
        self.adversarial = self.adversarial_metric.compute()
        self.adversarial_transfer = self.adversarial_transfer_metric.compute()
