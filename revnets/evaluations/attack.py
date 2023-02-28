from dataclasses import dataclass

import torch
from cacher.caches.speedup_deep_learning import cache

from ..data.mnist1d import Dataset
from ..networks.models.metrics import Metrics
from ..utils import config
from ..utils.trainer import Trainer
from .attack_model import AttackModel


@dataclass
class Evaluation:
    test: Metrics
    adversarial: Metrics
    adversarial_transfer: Metrics


def evaluate(
    original: torch.nn.Module, reconstruction: torch.nn.Module, network
) -> Evaluation:
    dataset: Dataset = network.dataset()
    return compare_attacks(original, reconstruction, dataset, config)


@cache
def compare_attacks(original, reconstruction, dataset: Dataset, _):
    # config argument passed to determine cache entry
    dataset.setup("valid")
    model = AttackModel(original, reconstruction)
    dataset.calibrate(model)
    dataloader = dataset.test_dataloader()
    Trainer().test(model, dataloaders=dataloader)
    return Evaluation(model.test, model.adversarial, model.adversarial_transfer)
