import torch
from torch.utils.data import Dataset

from ..utils import config


def split_train_val(dataset: Dataset, val_fraction=None) -> tuple[Dataset, ...]:
    if val_fraction is None:
        val_fraction = 0.1
    # ignore len(dataset) warning
    total_size = len(dataset)  # noqa
    val_size = int(val_fraction * total_size)
    train_size = total_size - val_size
    sizes = [train_size, val_size]
    # Make split deterministic for reproducibility
    split_generator = torch.Generator().manual_seed(config.manual_seed)
    train_data, val_data = torch.utils.data.random_split(
        dataset, sizes, generator=split_generator
    )
    return train_data, val_data
