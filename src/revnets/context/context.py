import time
from functools import cached_property

from package_utils.context import Context as Context_
from pytorch_lightning import Trainer

from ..models import Config, HyperParameters, Options, Path, Secrets
from ..utils import pl_logger


class Context(Context_[Options, Config, Secrets]):
    def __post_init__(self) -> None:
        experiment = self.options.experiment_name
        if experiment is not None and "analysis" in experiment:
            self.config.always_train = False
        if self.is_running_in_notebook:
            self.config.quiet_prediction = False
        if self.options.seed is not None:
            self.config.manual_seed = self.options.seed

    @property
    def epochs(self) -> int:
        return self.config.debug_epochs if self.config.debug else self.config.epochs

    @property
    def batch_size(self) -> int:
        return (
            self.config.debug_batch_size
            if self.config.debug
            else self.config.batch_size
        )

    @property
    def hyper_parameters(self) -> HyperParameters:
        return HyperParameters(
            self.epochs, self.config.learning_rate, self.config.bias_learning_rate
        )

    @property
    def number_of_devices(self) -> int:
        with pl_logger.Quiet():
            available_devices = Trainer(accelerator="auto").num_devices
        desired_number_of_devices = self.config.devices
        return min(desired_number_of_devices, available_devices)

    @cached_property
    def network_seed(self) -> int:
        seed = self.config.network_seed
        if self.config.randomize_training:
            now = time.time()
            self.config.manual_seed = int(now * 10**7) % 2**32
            if self.config.randomize_target:
                seed = self.config.manual_seed
        return seed

    @cached_property
    def is_running_in_notebook(self):
        try:
            get_ipython()
            is_in_notebook = True
        except NameError:
            is_in_notebook = False
        return is_in_notebook

    @property
    def log_folder(self) -> Path:
        return Path.outputs / self.options.experiment_name

    @property
    def save_dir(self) -> str:
        return str(self.log_folder)

    @property
    def test_results(self) -> str:
        return self.test_results_path.text

    @property
    def test_results_path(self) -> Path:
        return self.log_folder / "test_results.txt"


context = Context(Options, Config, Secrets)
