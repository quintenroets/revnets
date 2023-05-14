from typing import Any

from torch import nn, optim

from revnets.evaluations import weights
from revnets.networks import Network
from revnets.utils import config

from ...evaluations.weights.mae import Evaluator
from ...networks.models import trainable
from .metrics import Metrics


class ReconstructModel(trainable.Model):
    def __init__(
        self,
        model,
        network,
        visualize=False,
        visualization_interval: int = 10,
        clip_value: float = 10,
    ):
        super().__init__(model)
        self.network: Network = network
        self.visualization_interval = visualization_interval
        self.visualizer = weights.visualizer.Evaluator(self.model, self.network)
        self.do_visualize = visualize
        self.visualize(before_training=True)
        self.clip_value = clip_value
        self.losses = []
        self.lr = config.lr
        self.bias_lr = config.bias_lr or self.lr
        # self.automatic_optimization = False

    def log(self, *args, sync_dist=True, on_epoch=False, on_step=True, **kwargs):
        if "MAE" in args[0]:
            on_step = False
            on_epoch = True
        super().log(
            *args, sync_dist=sync_dist, on_epoch=on_epoch, on_step=on_step, **kwargs
        )

    def calculate_metrics(self, outputs, targets):
        return Metrics(
            l1_loss=nn.functional.l1_loss(outputs, targets),
            l2_loss=nn.functional.mse_loss(outputs, targets),
            # smooth_l1_loss=nn.functional.smooth_l1_loss(outputs, targets)
        )

    def training_step(self, batch, batch_idx):
        metrics: Metrics = self.obtain_metrics(batch, trainable.Phase.TRAIN)
        self.clip_weights()
        return metrics.loss
        self.manual_backward(metrics.loss)
        for optimizer in self.optimizers():
            optimizer.step()
            optimizer.zero_grad()

    def clip_weights(self):
        for param in self.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.log_weights_MAE()

    def log_weights_MAE(self) -> None:
        # pprint(self.model.state_dict())
        evaluator = Evaluator(self.model, self.network)
        state_dict = self.model.state_dict()
        mae = evaluator.evaluate()
        # undo standardize operations on state dict
        self.model.load_state_dict(state_dict)
        if mae is not None:
            self.log_weights_MAE_value(mae)

    def log_weights_MAE_value(self, mae):
        scales = {"": 0, "milli ": 3, "micro ": 6, "nano ": 9}
        scale_name = ""
        log_scale = 0
        for scale_name, log_scale in scales.items():
            scale = 10 ** (-log_scale)
            if mae > scale:
                break

        mae = mae * 10**log_scale

        self.log(f"weights {scale_name}MAE", mae, prog_bar=True)

    def visualize(self, before_training=False):
        if self.do_visualize:
            should_visualize = before_training or self.is_visualization_epoch
            if should_visualize:
                self.visualizer.evaluate()

    @property
    def is_visualization_epoch(self):
        return (self.trainer.current_epoch + 1) % self.visualization_interval == 0

    def weight_parameters(self):
        for name, parameter in self.named_parameters():
            param_type = name.split(".")[-1]
            if param_type == "weight":
                yield parameter

    def bias_parameters(self):
        for name, parameter in self.named_parameters():
            param_type = name.split(".")[-1]
            if param_type == "bias":
                yield parameter

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)

        weight_parameters = self.weight_parameters()
        weight_optimizer = optim.Adam(weight_parameters, lr=self.lr)
        bias_parameters = self.bias_parameters()
        bias_optimizer = optim.Adam(bias_parameters, lr=self.bias_lr)
        return weight_optimizer, bias_optimizer
