import enum
from dataclasses import asdict, dataclass
from functools import cached_property

import cli
from pytorch_lightning import Trainer

from . import pl_logger
from .args import get_args
from .logger import get_logger
from .path import Path
from .rank import rank_zero_only


@dataclass
class HyperParams:
    epochs: int
    lr: float


class Enum(enum.Enum):
    @classmethod
    def from_str(cls, label):
        return cls._value2member_map_[label]


class NetworkName(Enum):
    mininet = "mininet"
    mininet_untrained = "mininet_untrained"
    mediumnet = "mediumnet"
    mediumnet_untrained = "mediumnet_untrained"


@dataclass
class Config:
    network: str | tuple[str]
    config_path: Path
    epochs: int = None
    num_workers: int = 4
    lr: float = 0.1
    manual_seed: int = 77
    debug: bool = False
    debug_batch_limit: int = 1
    debug_batch_size: int = 16
    debug_epochs: int = 3
    devices: int = None
    batch_size: int = 128
    reconstruction_batch_size: int = batch_size
    log: bool = False
    log_on_debug: bool = False
    _num_devices: int = None
    adversarial_epsilon: float = 0.1
    visualize_attack: bool = False

    def __post_init__(self):
        if isinstance(self.network, str):
            self.network = (self.network,)
        self.network_names = [NetworkName.from_str(network) for network in self.network]

        if self.debug:
            self.epochs = self.debug_epochs
            self.batch_size = self.debug_batch_size
            self.always_do_train = True

        self.hyper_parameters = HyperParams(self.epochs, self.lr)

    @property
    def num_devices(self):
        return self._num_devices or self.calculated_num_devices

    @property
    def calculated_num_devices(self):
        available_devices = get_num_devices()
        return (
            available_devices
            if self.devices is None
            else min(self.devices, available_devices)
        )

    def dict(self):
        return asdict(self)

    @classmethod
    def load(cls):
        args = get_args()
        if not args.config_name:
            args.config_name = "baseline"
        config_path = (Path.config / args.config_name).with_suffix(".yaml")
        config = Config(config_path=config_path, **config_path.yaml)
        return config

    @property
    def network_print_name(self):
        return "_".join(network_name.value for network_name in self.network_names)

    def __repr__(self):
        return (
            f"network {self.network_print_name} with manual seed {self.manual_seed}\n"
        )

    def name(self):
        return f"{self.network_print_name}_seed{self.manual_seed}"

    @property
    def base_name(self):
        return f"{self.network_print_name}_seed{self.manual_seed}"

    @property
    def log_folder(self):
        return Path.outputs / self.base_name

    @property
    def save_dir(self):
        return str(self.log_folder)

    @rank_zero_only
    def show(self):
        cli.console.rule(f"[bold #000000]{self}")
        self.config_table.show()
        self.config_path.copy_to(self.log_folder / "config.yaml")

    @property
    def config_table(self):
        from ..utils.table import Table  # noqa: autoimport

        table = Table(show_lines=True)
        table.add_column("Configuration", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        config = self.config_path.yaml
        for name, value in config.items():
            # only show debug config in debug mode
            if self.debug or "debug" not in name:
                table.add_row(name, str(value))

        return table

    @property
    def limit_batches(self):
        return self.debug_batch_limit if self.debug else None

    @cached_property
    def logger(self):
        return get_logger(self)

    @property
    def test_results(self):
        return self.test_results_path.text

    @property
    def test_results_path(self):
        return self.log_folder / "test_results.txt"

    @rank_zero_only
    def show_test_results(self):
        cli.console.print(self.test_results, highlight=False)

    @property
    def num_sanity_val_steps(self):
        return 0 if self.debug else 2


# @cache
def get_num_devices():
    with pl_logger.Quiet():
        return Trainer(accelerator="auto").num_devices


config = Config.load()
