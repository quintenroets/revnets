from dataclasses import dataclass, field
from enum import Enum

from package_utils.dataclasses.mixins import SerializationMixin

from .experiment import Experiment


class Activation(Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    tanh = "tanh"


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float
    epochs: int = 1
    bias_learning_rate: float | None = None
    batch_size: int = 1
    activation: Activation = Activation.leaky_relu


@dataclass
class Evaluation:
    adversarial_epsilon: float = 0.1
    visualize_attack: bool = False
    use_align: bool = False


@dataclass
class Config(SerializationMixin):
    experiment: Experiment = field(default_factory=Experiment)
    num_workers: int = 8
    target_network_training: HyperParameters = HyperParameters(
        epochs=100, learning_rate=1.0e-2, batch_size=32
    )
    reconstruction_training: HyperParameters = HyperParameters(
        epochs=10, learning_rate=0.1, batch_size=256
    )
    reconstruction_training_debug: HyperParameters = HyperParameters(
        epochs=3, learning_rate=0.1, batch_size=16
    )
    evaluation: Evaluation = field(default_factory=Evaluation)

    early_stopping_patience: int = 100
    sampling_data_size: int = 10000
    n_rounds: int = 2
    n_networks: int = 2
    weight_variance_downscale_factor: float | None = None
    start_reconstruction_with_zero_biases: bool = False
    gradient_clip_val: int | None = None

    debug: bool = False
    debug_batch_limit: int = 1
    quiet_prediction: bool = True
    run_analysis: bool = False

    device: int = 1
    precision: int = 64
    always_train: bool = False
    loss_criterion: str = "l1"
    validation_ratio: float = 0

    console_metrics_refresh_interval: float = 0.5

    @property
    def number_of_validation_sanity_steps(self) -> int | None:
        return 0 if self.debug else 0

    @property
    def limit_batches(self) -> int | None:
        return self.debug_batch_limit if self.debug else None
