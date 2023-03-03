from dataclasses import asdict, dataclass, fields


@dataclass
class Evaluation:
    weights_MSE: str = None
    weights_MAE: str = None
    weights_MAE_layers: str = None
    train_outputs_MSE: str = None
    val_outputs_MSE: str = None
    test_outputs_MSE: str = None
    test_acc: str = None
    adversarial_test_acc: str = None
    adversarial_transfer_test_acc: str = None

    def metric_names(self):
        return [
            self.format_name(field.name)
            for field in fields(self)
            if self.dict()[field.name] is not None
        ]

    @classmethod
    def format_name(cls, name):
        name = name.replace("_", " ")
        return name[0].upper() + name[1:]

    def dict(self):
        return asdict(self)

    def values(self):
        for value in self.dict().values():
            if value is not None:
                yield value
