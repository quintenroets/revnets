from dataclasses import asdict, dataclass, fields


@dataclass
class Evaluation:
    weights_MSE: str = None
    weights_MAE: str = None
    weights_MAE_layers: str = None
    train_outputs_MAE: str = None
    val_outputs_MAE: str = None
    test_outputs_MAE: str = None
    test_acc: str = None
    adversarial_test_acc: str = None
    adversarial_transfer_test_acc: str = None

    def metric_names(self):
        valid_keys = self.dict().keys()
        return [
            self.format_name(field.name)
            for field in fields(self)
            if field.name in valid_keys
        ]

    @classmethod
    def format_name(cls, name):
        name = name.replace("_", " ")
        return name[0].upper() + name[1:]

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def values(self):
        yield from self.dict().values()
