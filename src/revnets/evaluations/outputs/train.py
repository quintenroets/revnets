import torch
from torch.utils.data import DataLoader

from revnets.data import Dataset

from . import val


class Evaluator(val.Evaluator):
    @classmethod
    def get_dataloader(cls, dataset: Dataset) -> DataLoader[tuple[torch.Tensor, ...]]:
        return dataset.train_dataloader(shuffle=False)
