from enum import Enum


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SILENT = "silent"
