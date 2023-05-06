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


@dataclass
class Config:
    config_path: Path
    epochs: int = None
    num_workers: int = 8
    lr: float = 0.1
    manual_seed: int = 77
    debug: bool = False
    debug_batch_limit: int = 1
    debug_batch_size: int = 16
    debug_epochs: int = 3
    devices: int = 1
    batch_size: int = 128
    reconstruction_batch_size: int = batch_size
    log: bool = False
    log_on_debug: bool = False
    _num_devices: int = None
    adversarial_epsilon: float = 0.1
    visualize_attack: bool = False
    run_analysis: bool = False
    precision: int = 64
    always_train: bool = None
    sampling_data_size: int = 10000
    quiet_prediction: bool = False
    randomize_training: bool = False
    n_rounds: int = None
    n_networks: int = 2
    weight_variance_downscale_factor: float = None

    def __post_init__(self):
        if self.debug:
            self.epochs = self.debug_epochs
            self.batch_size = self.debug_batch_size

        self.hyper_parameters = HyperParams(self.epochs, self.lr)
        self._num_devices = self.devices

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
            args.config_name = "config"
        config_path = (Path.config / args.config_name).with_suffix(".yaml")
        config_values = Config(config_path=config_path, **config_path.yaml)
        if args.experiment is not None and "analysis" in args.experiment:
            config_values.always_train = False
        return config_values

    def __repr__(self):
        return f"{self.name}\n"

    @property
    def name(self):
        return self.base_name

    @property
    def base_name(self):
        return f"seed_{self.manual_seed}"

    @property
    def log_folder(self):
        return Path.outputs / self.base_name

    @property
    def save_dir(self):
        return str(self.log_folder)

    @rank_zero_only
    def show(self):
        cli.console.rule("[bold #000000]Config")
        self.config_table.show()

    @property
    def config_table(self):
        from ..utils.table import Table  # noqa: autoimport

        table = Table(show_lines=True)
        table.add_column("Configuration", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        config_content = self.config_path.yaml
        for name, value in config_content.items():
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
        return 0 if self.debug else None


# @cache
def get_num_devices():
    with pl_logger.Quiet():
        return Trainer(accelerator="auto").num_devices


config = Config.load()
