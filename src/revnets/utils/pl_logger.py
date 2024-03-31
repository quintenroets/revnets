import logging
import warnings
from collections.abc import Sequence
from types import TracebackType


class Quiet:
    def __init__(self, names: Sequence[str] | None = None) -> None:
        if names is None:
            names = ("pytorch_lightning", "lightning")
        self.names = names
        self.old_loglevel = None

    def __enter__(self) -> None:
        self.old_log_levels = [
            logging.getLogger(name).getEffectiveLevel() for name in self.names
        ]
        for name in self.names:
            logger = logging.getLogger(name)
            logger.setLevel(0)
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
        # always keep level 0
