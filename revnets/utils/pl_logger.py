import logging


class Quiet:
    def __init__(self, name="pytorch_lightning.utilities.rank_zero"):
        self.name = name
        self.old_loglevel = None

    def __enter__(self):
        self.old_loglevel = logging.getLogger(self.name).getEffectiveLevel()
        logging.getLogger(self.name).setLevel(logging.WARNING)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger(self.name).setLevel(self.old_loglevel)
