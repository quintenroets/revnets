from dataclasses import dataclass, field

from torchvision import datasets, transforms

from revnets.models import Path

from . import base


@dataclass
class DataModule(base.DataModule):
    path: str = str(Path.data / "mnist")
    transformation: transforms.Compose = field(
        default_factory=lambda: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        ),
    )

    def prepare_data(self) -> None:
        for train in (False, True):
            datasets.MNIST(self.path, train=train, download=True)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self.train_validation = self.load_dataset(train=True)
        self.test = self.load_dataset(train=False)
        self.split_train_validation()

    def load_dataset(self, *, train: bool) -> datasets.MNIST:
        return datasets.MNIST(
            self.path,
            train=train,
            download=True,
            transform=self.transformation,
        )
