from dataclasses import dataclass
from enum import Enum

from superpathlib import Path


@dataclass
class HyperParameters:
    epochs: int
    learning_rate: float
    bias_learning_rate: float


class Activation(Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    tanh = "tanh"


@dataclass
class Config:
    output_path: Path | None = None
    secrets_path: Path | None = None
    num_workers: int = 8

    blackbox_batch_size: int = 32
    blackbox_learning_rate: float = 1.0e-2
    blackbox_epochs: int = 100

    batch_size: int = 256
    learning_rate: float = 0.1
    bias_learning_rate: float = None
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
    activation: Enum = Activation.leaky_relu

    console_metrics_refresh_interval: float = 0.5

    @property
    def number_of_validation_sanity_steps(self) -> int:
        return 0 if self.debug else None

    @property
    def limit_batches(self) -> int | None:
        return self.debug_batch_limit if self.debug else None
