from functools import cached_property

import torch
from package_utils.context import Context as Context_

from ..models import Config, Options, Path


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
        path = self.output_path / "results.yaml"
        return path.with_nonexistent_name()

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
                raise ValueError(f"Unsupported precision {self.config.precision}")
        return dtype


context = Context(Options, Config, None)
