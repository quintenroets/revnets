from ..context import context
from ..models import Experiment
from .lauch_plan import LaunchPlan


def main() -> None:
    """
    Launch experiments.
    """
    launch_plan = LaunchPlan()
    for experiment in launch_plan.experiments_to_launch():
        launch(experiment)


def launch(experiment: Experiment) -> None:
    experiment.prepare_config(context.options.config_path.yaml)
    print(experiment.config_path)
    cli.run("ls")
    # cli.run("revnets", "--config-path", experiment.config_path)
