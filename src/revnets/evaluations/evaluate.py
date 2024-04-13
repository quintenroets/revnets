from types import ModuleType
from typing import Any

from torch.nn import Module

from revnets.context import context

from ..pipelines import Pipeline
from . import analysis, outputs, weights
from .base import Evaluator
from .evaluation import Evaluation


def evaluate(reconstruction: Module, pipeline: Pipeline) -> Evaluation:
    def apply(evaluation_module: ModuleType) -> Any:
        evaluator: Evaluator = evaluation_module.Evaluator(reconstruction, pipeline)
        return evaluator.get_evaluation()

    analysis_modules = (analysis.weights,)

    if context.config.evaluation.run_analysis:
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
