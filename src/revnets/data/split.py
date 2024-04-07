from enum import Enum


class Split(Enum):
    train = "train"
    valid = "valid"
    test = "test"
    train_val = "train_val"

    @property
    def is_train(self) -> bool:
        return self == Split.train

    @property
    def is_valid(self) -> bool:
        return self == Split.valid

    @property
    def is_train_or_valid(self) -> bool:
        return self.is_train or self.is_valid
