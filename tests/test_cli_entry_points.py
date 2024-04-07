from unittest.mock import MagicMock, patch

from package_dev_utils.tests.args import no_cli_args
from revnets.cli import entry_point, launch


@no_cli_args
@patch("revnets.main.main.main")
def test_main(_: MagicMock) -> None:
    entry_point.entry_point()


@no_cli_args
@patch("cli.run")
def test_launch(_: MagicMock) -> None:
    launch.entry_point()
