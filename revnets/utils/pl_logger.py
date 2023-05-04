import logging
import warnings


class Quiet:
    def __init__(self, names=None):
        if names is None:
            names = ("pytorch_lightning", "lightning")
        self.names = names
        self.old_loglevel = None

    def __enter__(self):
        self.old_log_levels = [
            logging.getLogger(name).getEffectiveLevel() for name in self.names
        ]
        for name in self.names:
            logger = logging.getLogger(name)
            logger.setLevel(0)
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # always keep level 0
