from revnets.utils import config

from . import logging
from .metrics import Metrics


class ReconstructModel(logging.ReconstructModel):
    def __init__(
        self, *args, clip_value: float = 10, use_loss_sum: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_value = clip_value
        self.learning_rate_scale = (
            config.batch_size * self.network.output_size if use_loss_sum else 1
        )
        self._learning_rate /= self.learning_rate_scale
        self.use_loss_sum = use_loss_sum
        self.val = 0

    def clip_weights(self) -> None:
        for param in self.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

    def on_train_epoch_end(self) -> None:
        self.clip_weights()
        super().on_train_epoch_end()

    def extract_loss(self, metrics: Metrics):
        return metrics.loss_sum if self.use_loss_sum else metrics.loss

    @property
    def learning_rate(self):
        return self._learning_rate * self.learning_rate_scale

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        parent_property = super(ReconstructModel, ReconstructModel).learning_rate
        parent_property.__set__(self, value / self.learning_rate_scale)
