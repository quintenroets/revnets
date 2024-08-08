import math
from dataclasses import dataclass
from types import ModuleType
from typing import cast

import pytest
from revnets import evaluations, pipelines, reconstructions
from revnets.context import Context
from revnets.evaluations import analysis, attack, outputs, weights
from revnets.evaluations.evaluate import format_percentage
from revnets.networks import mininet
from revnets.pipelines import Pipeline
from torch.nn import Sequential

from tests.evaluations import verifier
from tests.evaluations.verifier import Standardization

weight_evaluation_modules = (
    weights.mse,
    weights.mae,
    weights.max_ae,
    weights.layers_mae,
    weights.named_layers_mae,
)

evaluation_modules = (
    *weight_evaluation_modules,
    attack.attack,
    outputs.train,
    outputs.validation,
    outputs.test,
    analysis.weights,
    analysis.activations,
    analysis.trained_target,
)

pipeline_modules = (
    pipelines.mininet,
    pipelines.images.rnn,
)


@pytest.fixture()
def pipeline() -> Pipeline:
    return pipelines.mininet.Pipeline()


def test_cheat_evaluations(pipeline: Pipeline) -> None:
    reconstructor = reconstructions.cheat.Reconstructor(pipeline)
    reconstruction = reconstructor.create_reconstruction()

    evaluation_metrics = evaluations.evaluate(reconstruction, pipeline)
    # cheat should give perfect metrics
    perfect_metrics = (
        evaluation_metrics.weights_MAE,
        evaluation_metrics.train_outputs_MAE,
        evaluation_metrics.val_outputs_MAE,
        evaluation_metrics.train_outputs_MAE,
    )
    for value in perfect_metrics:
        if value is not None and value != "/":
            assert math.isclose(float(value), 0, abs_tol=1e-5)


@pytest.mark.parametrize("evaluation_module", evaluation_modules)
@pytest.mark.parametrize("pipeline_module", pipeline_modules)
@pytest.mark.usefixtures("test_context")
def test_evaluations(
    evaluation_module: ModuleType,
    pipeline_module: ModuleType,
) -> None:
    pipeline = cast(Pipeline, pipeline_module.Pipeline())
    reconstructor = reconstructions.empty.Reconstructor(pipeline)
    reconstruction = reconstructor.create_reconstruction()
    evaluation_module.Evaluator(reconstruction, pipeline).get_evaluation()


@dataclass
class Verifier(verifier.Verifier):
    network: Sequential
    target: Sequential

    def __post_init__(self) -> None:
        pass


@pytest.mark.parametrize("evaluation_module", weight_evaluation_modules)
def test_weight_evaluations(
    evaluation_module: ModuleType,
    pipeline: Pipeline,
    test_context: Context,
) -> None:
    reconstructor = reconstructions.empty.Reconstructor(pipeline)
    reconstruction = reconstructor.create_reconstruction()
    evaluator: weights.Evaluator = evaluation_module.Evaluator(reconstruction, pipeline)
    evaluator.evaluate()
    verify_evaluated_networks(evaluator, pipeline, test_context)


def verify_evaluated_networks(
    evaluator: weights.Evaluator,
    pipeline: Pipeline,
    context: Context,
) -> None:
    use_align = context.config.evaluation.use_align
    standardization = (
        Standardization.align if use_align else Standardization.standardize
    )
    for network in (evaluator.target, evaluator.reconstruction):
        network_verifier = Verifier(
            network_module=mininet,
            standardization_type=standardization,
            activation=pipeline.network_factory.activation,
            network=cast(Sequential, network),
            target=pipeline.target,
        )
        network_verifier.verify_form()


def test_format_percentage() -> None:
    format_percentage(0.1)
