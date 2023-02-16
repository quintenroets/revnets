from dataclasses import asdict, dataclass, fields

from . import outputs, weights


@dataclass
class Evaluation:
    weights_mse: float
    outputs_mse: float

    @classmethod
    def metric_names(cls):
        return [field.name for field in fields(cls)]

    def dict(self):
        return asdict(self)

    def get_value_list(self):
        precision = 3
        values = self.dict().values()
        values = [f"{v:.{precision}f}" for v in values]
        return values


def evaluate(original, reconstruction, network, *_, **__):
    return Evaluation(
        weights_mse=weights.evaluate(original, reconstruction),
        outputs_mse=outputs.evaluate(original, reconstruction, network),
    )
