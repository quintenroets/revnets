from dataclasses import asdict, dataclass, fields


@dataclass
class Evaluation:
    weights_mse: float
    train_outputs_mse: float
    val_outputs_mse: float
    test_outputs_mse: float
    test_acc: float
    adversarial_test_acc: float
    adversarial_transfer_test_acc: float

    @classmethod
    def metric_names(cls):
        return [field.name.replace("_", " ") for field in fields(cls)]

    def dict(self):
        return asdict(self)

    def get_value_list(self):
        precision = 3
        values = self.dict().values()
        values = ["/" if v is None else f"{v:.{precision}f}" for v in values]
        return values
