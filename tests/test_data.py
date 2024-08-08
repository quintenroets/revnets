from types import ModuleType
from typing import TYPE_CHECKING

import pytest
from revnets import data

if TYPE_CHECKING:
    from revnets.data import DataModule  # pragma: nocover

dataset_modules = data.mnist, data.mnist1d


@pytest.mark.parametrize("dataset_module", dataset_modules)
def test_setup(dataset_module: ModuleType) -> None:
    dataset: DataModule = dataset_module.DataModule(batch_size=10)
    dataset.prepare_data()
    dataset.setup("train")
    dataloaders = (
        dataset.train_dataloader(),
        dataset.val_dataloader(),
        dataset.test_dataloader(),
    )
    for dataloader in dataloaders:
        item = next(iter(dataloader))
        assert item is not None
