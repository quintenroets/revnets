import logging
import warnings


class Quiet:
    def __init__(self, name="pytorch_lightning"):
        self.name = name
        self.old_loglevel = None

    def __enter__(self):
        self.old_loglevel = logging.getLogger(self.name).getEffectiveLevel()
        logger = logging.getLogger(self.name)
        logger.setLevel(0)
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # always keep level 0
