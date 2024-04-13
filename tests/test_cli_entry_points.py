from unittest.mock import MagicMock, patch

from package_dev_utils.tests.args import no_cli_args
from revnets.cli import entry_point, launch


@no_cli_args
@patch("revnets.main.main.Experiment.run")
def test_main(run: MagicMock) -> None:
    entry_point.entry_point()
    run.assert_called_once()


@no_cli_args
@patch("cli.run")
def test_launch(run: MagicMock) -> None:
    launch.entry_point()
    run.assert_called()
