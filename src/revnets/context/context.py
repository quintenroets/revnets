from functools import cached_property
from typing import cast

<<<<<<< HEAD
import torch
from package_utils.context import Context as Context_
=======
from .config import Config
from .options import Options
from .secrets_ import Secrets
>>>>>>> template

from revnets.models import Config, Options, Path


class Context(Context_[Options, Config, None]):
    @cached_property
    def is_running_in_notebook(self) -> bool:
        try:
            get_ipython()  # type: ignore[name-defined]
            is_in_notebook = True  # pragma: nocover
        except NameError:
            is_in_notebook = False
        return is_in_notebook

    @property
    def output_path(self) -> Path:
        return self.config.experiment.path

    @property
    def results_path(self) -> Path:
        relative_path = self.output_path.relative_to(Path.config)
        path = Path.results / relative_path / "results.yaml"
        path = path.with_nonexistent_name()
        return cast("Path", path)

    @property
    def log_path(self) -> Path:
        return self.output_path / "logs"

    @property
    def log_path_str(self) -> str:
        return str(self.log_path)

    @cached_property
    def device(self) -> torch.device:
        name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)

    @property
    def dtype(self) -> torch.dtype:
        match self.config.precision:  # pragma: nocover
            case 32:
                dtype = torch.float32
            case 64:
                dtype = torch.float64
            case _:
                message = f"Unsupported precision {self.config.precision}"
                raise ValueError(message)
        return dtype


context = Context(Options, Config, None)
