from typing import TypeVar

import torch
from torch.utils.data import Dataset

from ..context import context

T = TypeVar("T")


def split_train_val(
    dataset: Dataset[T], val_fraction: float | None = None
) -> tuple[Dataset[T], ...]:
    if val_fraction is None:
        val_fraction = 0.1
    # ignore len(dataset) warning
    total_size = len(dataset)  # type: ignore[arg-type]
    val_size = int(val_fraction * total_size)
    train_size = total_size - val_size
    sizes = [train_size, val_size]
    # Make split deterministic for reproducibility
    seed = context.config.experiment.target_network_seed
    split_generator = torch.Generator().manual_seed(seed)
    train_data, val_data = torch.utils.data.random_split(
        dataset, sizes, generator=split_generator
    )
    return train_data, val_data
