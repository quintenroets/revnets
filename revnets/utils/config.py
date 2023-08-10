from __future__ import annotations

import enum
import time
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
    bias_lr: float


class Enum(enum.Enum):
    @classmethod
    def from_str(cls, name):
        return cls._value2member_map_[name]


class Activation(Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    tanh = "tanh"


@dataclass
class Config:
    config_path: Path
    num_workers: int = 8

    blackbox_batch_size: int = 32
    blackbox_lr: float = 1.0e-2
    blackbox_epochs: int = 100

    batch_size: int = 256
    lr: float = 0.1
    bias_lr: float = None
    epochs: int = None
    early_stopping_patience: int = 100

    sampling_data_size: int = 10000
    n_rounds: int = None
    n_networks: int = 2
    weight_variance_downscale_factor: float = None
    gradient_clip_val: int = None

    manual_seed: int = 77
    network_seed: int = 77
    randomize_training: bool = False
    randomize_target: bool = False
    debug: bool = False
    debug_batch_limit: int = 1
    debug_batch_size: int = 16
    debug_epochs: int = 3
    log: bool = False
    log_on_debug: bool = False
    quiet_prediction: bool = True
    run_analysis: bool = False

    devices: int = 1
    _num_devices: int = None

    adversarial_epsilon: float = 0.1
    visualize_attack: bool = False

    precision: int = 64
    always_train: bool = None
    use_align: bool = True
    loss_criterion: str = "l1"
    validation_ratio: float = 0
    activation: str | Enum = "leaky_relu"

    console_metrics_refresh_interval: float = 0.5

    def __post_init__(self):
        self.sampling_data_size = int(self.sampling_data_size)
        self.lr = float(self.lr)
        self.blackbox_lr = float(self.blackbox_lr)
        if self.bias_lr is not None:
            self.bias_lr = float(self.bias_lr)
        self.sampling_data_size = int(self.sampling_data_size)

        if self.debug:
            self.epochs = self.debug_epochs
            self.batch_size = self.debug_batch_size

        self.hyper_parameters = HyperParams(self.epochs, self.lr, self.bias_lr)
        self._num_devices = self.devices

        if self.randomize_training:
            now = time.time()
            self.manual_seed = int(now * 10**7) % 2**32
            if self.randomize_target:
                self.network_seed = self.manual_seed

        self.activation = Activation.from_str(self.activation)

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
        if config_values.is_notebook:
            config_values.quiet_prediction = False
        if args.seed is not None:
            config_values.manual_seed = int(args.seed)
        return config_values

    @property
    def is_notebook(self):
        try:
            get_ipython()
            is_in_notebook = True
        except NameError:
            is_in_notebook = False
        return is_in_notebook

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
