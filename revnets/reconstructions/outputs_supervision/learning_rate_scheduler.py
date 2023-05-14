from revnets.utils import config

from . import model


class ReconstructModel(model.ReconstructModel):
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.losses.append(self.logged_loss)
        self.check_learning_rate()

    def check_learning_rate(self):
        target_learning_rate = config.lr
        if self.logged_loss < 0.01:
            target_learning_rate = config.lr / 1000
        if self.logged_loss < 0.02:
            target_learning_rate = config.lr / 500
        elif self.logged_loss < 0.1:  # was 0.1
            target_learning_rate = config.lr / 100
        elif self.logged_loss < 0.2:  # was 0.1
            target_learning_rate = config.lr / 50
        elif self.logged_loss < 0.3:  # was 0.1
            target_learning_rate = config.lr / 30
        elif self.logged_loss < 0.5:  # was 0.1
            target_learning_rate = config.lr / 20
        elif self.logged_loss < 1:
            target_learning_rate = config.lr / 10
        self.set_learning_rate(target_learning_rate)

    def set_learning_rate(self, learning_rate):
        optimizers = (self.optimizers(),)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                if param_group["lr"] > learning_rate:
                    old_lr = param_group["lr"]
                    message = f"Lowering learning rate from {old_lr} to {learning_rate}"
                    print(message)
                    param_group["lr"] = learning_rate

    @property
    def logged_loss(self):
        metrics = self.trainer.callback_metrics
        return metrics["train l1_loss"]
