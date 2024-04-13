import math
from types import ModuleType

import pytest
from revnets import evaluations, pipelines, reconstructions
from revnets.evaluations import analysis, attack, outputs, weights
from revnets.pipelines import Pipeline

evaluation_modules = (
    weights.mse,
    weights.mae,
    weights.max_ae,
    weights.layers_mae,
    weights.named_layers_mae,
    attack.attack,
    outputs.train,
    outputs.val,
    outputs.test,
    analysis.weights,
    analysis.activations,
    analysis.trained_target,
)


@pytest.fixture
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
def test_evaluations(
    evaluation_module: ModuleType, pipeline: Pipeline, test_context: None
) -> None:
    reconstructor = reconstructions.empty.Reconstructor(pipeline)
    reconstruction = reconstructor.create_reconstruction()
    evaluation_module.Evaluator(reconstruction, pipeline).evaluate()
