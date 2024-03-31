from .. import experiments
from ..context import context


def main() -> None:
    """
    Reverse engineer internal parameters of black box neural networks.
    """
    if context.options.seed_range:
        for seed in range(context.options.seed_range):
            context.config.manual_seed = seed
            run()
    else:
        run()


def run() -> None:
    experiment_module = get_experiment_module()
    experiment_module.Experiment().run()


def get_experiment_module():
    experiment = context.options.experiment_name
    if experiment is not None:
        experiment_module = experiments
        analysis_keyword = "_analysis"
        if analysis_keyword in experiment:
            experiment = experiment.replace(analysis_keyword, "")
            experiment_module = experiments.analysis
        experiment_module = getattr(experiment_module, experiment)
    else:
        experiment_module = experiments.experiment
    return experiment_module
