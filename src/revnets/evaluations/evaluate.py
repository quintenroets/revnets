from types import ModuleType

from torch.nn import Module

from ..pipelines import Pipeline
from . import analysis, weights
from .base import Evaluator
from .evaluation import Evaluation


def evaluate(reconstruction: Module, pipeline: Pipeline) -> Evaluation:
    def apply(evaluation_module: ModuleType) -> str:
        evaluator: Evaluator = evaluation_module.Evaluator(reconstruction, pipeline)
        return evaluator.get_evaluation()

    analysis_modules = (analysis.weights,)
    for analysis_module in analysis_modules:
        apply(analysis_module)

    # attack_evaluation = apply(attack)

    return Evaluation(
        weights_MSE=apply(weights.mse),
        weights_MAE=apply(weights.mae),
        weights_max_AE=apply(weights.max_ae),
        weights_MAE_layers=apply(weights.layers_mae),
        # test_outputs_MAE=apply(outputs.test).mae,
        # test_acc=format_percentage(attack_evaluation.test.accuracy),
        # adversarial_test_acc=
        # format_percentage(attack_evaluation.adversarial.accuracy),
        # adversarial_transfer_test_acc=format_percentage(
        # attack_evaluation.adversarial_transfer.accuracy
        # ),
    )


def format_percentage(value: float) -> str:
    percentage = value * 100
    return f"{percentage:.{1}f}%"
