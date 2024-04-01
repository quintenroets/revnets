from types import ModuleType

from .. import experiments
from ..context import context


def main() -> None:
    name = context.options.experiment
    module = experiments.experiment if name is None else extract_experiment_module(name)
    module.Experiment().run()


def extract_experiment_module(name: str) -> ModuleType:
    experiment_module = experiments
    analysis_keyword = "_analysis"
    if analysis_keyword in name:
        name = name.replace(analysis_keyword, "")
        experiment_module = experiments.analysis
    return getattr(experiment_module, name)
