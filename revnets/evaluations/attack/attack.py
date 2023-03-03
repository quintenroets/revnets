from dataclasses import dataclass

from cacher.caches.speedup_deep_learning import cache

from ...networks.models.metrics import Metrics
from ...utils import config
from ...utils.trainer import Trainer
from .. import base
from .attack_model import AttackModel


@dataclass
class Evaluation:
    test: Metrics
    adversarial: Metrics
    adversarial_transfer: Metrics


class Evaluator(base.Evaluator):
    def evaluate(self) -> Evaluation:
        return self.compare_attacks(config)

    @cache
    def compare_attacks(self, _):
        # config argument passed to determine cache entry
        dataset = self.get_dataset()
        dataset.setup("valid")
        model = AttackModel(self.original, self.reconstruction)
        dataset.calibrate(model)
        dataloader = dataset.test_dataloader()
        Trainer().test(model, dataloaders=dataloader)
        return Evaluation(model.test, model.adversarial, model.adversarial_transfer)
