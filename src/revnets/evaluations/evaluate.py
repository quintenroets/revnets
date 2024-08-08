from types import ModuleType
from typing import TYPE_CHECKING, Any

from torch.nn import Module

from revnets.context import context
from revnets.pipelines import Pipeline

from . import analysis, outputs, weights
from .evaluation import Evaluation

if TYPE_CHECKING:
    from .base import Evaluator  # pragma: nocover


def evaluate(reconstruction: Module, pipeline: Pipeline) -> Evaluation:
    def apply(evaluation_module: ModuleType) -> Any:
        evaluator: Evaluator = evaluation_module.Evaluator(reconstruction, pipeline)
        return evaluator.get_evaluation()

    if context.config.evaluation.run_analysis:
        analysis_modules = (analysis.weights,)
        for analysis_module in analysis_modules:
            apply(analysis_module)

    return Evaluation(
        weights_MSE=apply(weights.mse),
        weights_MAE=apply(weights.mae),
        weights_max_AE=apply(weights.max_ae),
        weights_MAE_layers=apply(weights.layers_mae),
        test_outputs_MAE=apply(outputs.test).mae,
    )


def format_percentage(value: float) -> str:
    percentage = value * 100
    return f"{percentage:.{1}f}%"
