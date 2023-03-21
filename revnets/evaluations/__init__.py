from types import ModuleType

from . import attack, outputs, weights
from .evaluation import Evaluation


def evaluate(original, reconstruction, network):
    def apply(evaluation_module: ModuleType):
        evaluator = evaluation_module.Evaluator(original, reconstruction, network)
        return evaluator.get_evaluation()

    attack_evaluation = apply(attack)

    return Evaluation(
        weights_MSE=apply(weights.mse),
        weights_MAE=apply(weights.mae),
        weights_MAE_layers=apply(weights.layers_mae),
        val_outputs_MSE=apply(outputs.val),
        test_acc=format_percentage(attack_evaluation.test.accuracy),
        adversarial_test_acc=format_percentage(attack_evaluation.adversarial.accuracy),
        adversarial_transfer_test_acc=format_percentage(
            attack_evaluation.adversarial_transfer.accuracy
        ),
    )


def format_percentage(value):
    percentage = value * 100
    return f"{percentage:.{1}f}%"
