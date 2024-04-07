from dataclasses import dataclass

import torch

from revnets.training import Metrics, Trainer

from .. import base
from .attack_model import AttackModel


@dataclass
class Evaluation:
    test: Metrics
    adversarial: Metrics
    adversarial_transfer: Metrics


class Evaluator(base.Evaluator):
    def evaluate(self) -> Evaluation:
        return self.compare_attacks()

    def compare_attacks(self):
        dataset = self.get_dataset()
        dataset.prepare()
        model = AttackModel(self.original, self.reconstruction)
        dataset.calibrate(model)
        dataloader = dataset.test_dataloader()
        precision = 32  # adversarial attack library only works with precision 32
        dtype = torch.float32
        for network in (model.reconstruction, model.original):
            network.to(dtype)
        Trainer(precision=precision).test(model, dataloaders=dataloader)
        return Evaluation(model.test, model.adversarial, model.adversarial_transfer)
