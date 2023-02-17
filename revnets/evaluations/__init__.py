from . import outputs_test, outputs_train, outputs_val, weights
from .evaluation import Evaluation


def evaluate(original, reconstruction, network, *_, **__):
    return Evaluation(
        weights_mse=weights.evaluate(original, reconstruction),
        train_outputs_mse=outputs_train.evaluate(original, reconstruction, network),
        val_outputs_mse=outputs_val.evaluate(original, reconstruction, network),
        test_outputs_mse=outputs_test.evaluate(original, reconstruction, network),
    )
