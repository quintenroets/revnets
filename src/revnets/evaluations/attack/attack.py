from dataclasses import dataclass
from typing import cast

import torch

from revnets.evaluations import base
from revnets.training import Trainer
from revnets.training.targets import Metrics

from .network import AttackNetwork


@dataclass
class Evaluation:
    test: Metrics
    adversarial: Metrics
    adversarial_transfer: Metrics


class Evaluator(base.Evaluator):
    def evaluate(self) -> Evaluation:
        return self.compare_attacks()

    def compare_attacks(self) -> Evaluation:
        data = self.pipeline.load_prepared_data()
        model = AttackNetwork(self.target, self.reconstruction)
        dataloader = data.test_dataloader()
        precision = 32  # adversarial attack library only works with precision 32
        dtype = torch.float32
        for network in (model.reconstruction, model.original):
            network.to(dtype)
        Trainer(precision=precision).test(model, dataloaders=dataloader)
        untyped_metrics = model.test, model.adversarial, model.adversarial_transfer
        metrics = cast(tuple[Metrics, ...], untyped_metrics)
        return Evaluation(*metrics)
