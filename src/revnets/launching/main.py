from typing import Any, cast

import cli

from revnets.context import context
from revnets.models import Experiment

from .lauch_plan import LaunchPlan


def main() -> None:
    """
    Launch experiments.
    """
    launch_plan = LaunchPlan()
    for experiment in launch_plan.experiments_to_launch():
        launch(experiment)


def launch(experiment: Experiment) -> None:
    config_dict = cast(dict[str, Any], context.options.config_path.yaml)
    experiment.prepare_config(config_dict)
    command = "revnets", "--config-path", str(experiment.config_path)
    title = "_".join(experiment.names)
    launch_command(command, title)


def launch_command(command: tuple[str, ...], title: str) -> None:
    command_str = " ".join(command)
    shell_command = f"{command_str}; fish"
    launch_command = ("tmux", "new-session", "-s", title, "-d", shell_command)
    cli.run(launch_command)
