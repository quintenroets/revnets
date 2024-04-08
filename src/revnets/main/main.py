from types import ModuleType
from typing import cast

from .. import experiments
from ..context import context


def main() -> None:
    name = context.options.experiment
    module = experiments.experiment if name is None else extract_experiment_module(name)
    module.Experiment().run()


def extract_experiment_module(name: str) -> ModuleType:
    module: ModuleType = experiments
    analysis_keyword = "_analysis"
    if analysis_keyword in name:
        name = name.replace(analysis_keyword, "")
        module = experiments.analysis
    experiment_module = getattr(module, name)
    return cast(ModuleType, experiment_module)
