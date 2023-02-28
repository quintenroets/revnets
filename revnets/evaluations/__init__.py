from . import attack, outputs_test, outputs_train, outputs_val
from .evaluation import Evaluation
from .weights import weights


def evaluate(original, reconstruction, network, *_, **__):
    attack_evaluation = attack.evaluate(original, reconstruction, network)

    return Evaluation(
        weights_mse=weights.evaluate(original, reconstruction),
        train_outputs_mse=outputs_train.evaluate(original, reconstruction, network),
        val_outputs_mse=outputs_val.evaluate(original, reconstruction, network),
        test_outputs_mse=outputs_test.evaluate(original, reconstruction, network),
        test_acc=attack_evaluation.test.accuracy,
        adversarial_test_acc=attack_evaluation.adversarial.accuracy,
        adversarial_transfer_test_acc=attack_evaluation.adversarial_transfer.accuracy,
    )
