from torchvision import datasets, transforms

from ..models import Path
from . import base


class Dataset(base.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.path = Path.data / "mnist"
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self) -> None:
        for train in (False, True):
            datasets.MNIST(self.path, train=train, download=True)

    def setup(self, stage: str | None = None) -> None:
        self.train_val_dataset = datasets.MNIST(
            self.path, train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            self.path, train=False, download=True, transform=self.transform
        )
        super().setup()
