from typing import Any

from torch.utils.data import DataLoader

from revnets.data import DataModule

from . import validation


class Evaluator(validation.Evaluator):
    @classmethod
    def extract_dataloader(cls, data: DataModule) -> DataLoader[Any]:
        return data.train_dataloader(shuffle=False)
