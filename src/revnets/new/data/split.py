from ..utils import Enum


class Split(Enum):
    train = "train"
    valid = "valid"
    test = "test"
    train_val = "train_val"

    @property
    def is_train(self):
        return self == Split.train

    @property
    def is_valid(self):
        return self == Split.valid

    @property
    def is_train_or_valid(self):
        return self.is_train or self.is_valid
