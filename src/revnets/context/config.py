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
    run_analysis: bool = False
    only_visualize_differences: bool = True


@dataclass
class Config(SerializationMixin):
    sampling_data_size: int = 102400
    reconstruction_training: HyperParameters = field(
        default_factory=lambda: HyperParameters(
            epochs=300,
            learning_rate=1e-2,
            batch_size=256,
        ),
    )
    reconstruct_from_checkpoint: bool = False
    always_train: bool = True
    n_rounds: int = 2
    experiment: Experiment = field(default_factory=Experiment)

    reconstruction_training_debug: HyperParameters = field(
        default_factory=lambda: HyperParameters(
            epochs=3,
            learning_rate=0.1,
            batch_size=16,
        ),
    )
    target_network_training: HyperParameters = field(
        default_factory=lambda: HyperParameters(
            epochs=100,
            learning_rate=1.0e-2,
            batch_size=32,
        ),
    )
    difficult_inputs_training: HyperParameters = field(
        default_factory=lambda: HyperParameters(
            epochs=1000,
            learning_rate=1.0e-3,
        ),
    )
    evaluation: Evaluation = field(default_factory=Evaluation)
    evaluation_batch_size: int = 1000

    num_workers: int = 8
    early_stopping_patience: int = 10
    n_networks: int = 2
    visualization_interval = 10
    weight_variance_downscale_factor: float | None = None
    start_reconstruction_with_zero_biases: bool = False
    gradient_clip_val: int | None = None

    debug: bool = False
    debug_batch_limit: int = 1
    quiet_prediction: bool = True

    device: int = 1
    precision: int = 64
    loss_criterion: str = "l1"
    validation_ratio: float = 0.1

    console_metrics_refresh_interval: float = 0.5

    limit_batches: int | None = None

    @property
    def number_of_validation_sanity_steps(self) -> int | None:
        return 0 if self.debug else 1

    def __post_init__(self) -> None:
        if self.evaluation.run_analysis:
            self.always_train = False
        if self.debug:
            self.limit_batches = self.debug_batch_limit  # pragma: nocover
