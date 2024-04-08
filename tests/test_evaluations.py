import math
from types import ModuleType

import pytest
from revnets import evaluations, pipelines, reconstructions

pipeline_modules = (pipelines.mininet,)


@pytest.mark.parametrize("pipeline_module", pipeline_modules)
def test_cheat_evaluations(pipeline_module: ModuleType) -> None:
    pipeline = pipeline_module.Pipeline()
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
