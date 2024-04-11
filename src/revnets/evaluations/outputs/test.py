from typing import Any

from torch.utils.data import DataLoader

from revnets.data import DataModule

from . import val


class Evaluator(val.Evaluator):
    @classmethod
    def extract_dataloader(cls, data: DataModule) -> DataLoader[Any]:
        return data.test_dataloader()
