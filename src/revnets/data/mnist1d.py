from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import requests
import torch
from numpy.typing import NDArray
from package_utils.dataclasses import SerializationMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ..models import Path
from . import base


@dataclass
class RawData(SerializationMixin):
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.float32]

    @classmethod
    def from_path(cls, path: Path) -> RawData:
        with path.open("rb") as fp:
            data = pickle.load(fp)
        return cls(data["x"], data["y"], data["x_test"], data["y_test"])

    def scale(self) -> None:
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.x_test = scaler.transform(self.x_test)

    def extract_train_validation(self) -> TensorDataset:
        x = torch.Tensor(self.x)
        y = torch.LongTensor(self.y)
        return TensorDataset(x, y)

    def extract_test(self) -> TensorDataset:
        x = torch.Tensor(self.x_test)
        y = torch.LongTensor(self.y_test)
        return TensorDataset(x, y)


@dataclass
class DataModule(base.DataModule):
    path: Path = Path.data / "mnist_1D"
    raw_path: Path = Path.data / "mnist_1D.pkl"
    download_url: str = (
        "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
    )

    def prepare_data(self) -> None:
        if not self.path.exists():
            if not self.raw_path.exists():
                self.download()
            self.process()

    def download(self) -> None:
        response = requests.get(self.download_url, allow_redirects=True)
        self.raw_path.byte_content = response.content

    def process(self) -> None:
        raw_data = RawData.from_path(self.raw_path)
        raw_data.scale()
        data = raw_data.extract_train_validation(), raw_data.extract_test()
        path = str(self.path)
        torch.save(data, path)

    def setup(self, stage: str) -> None:
        path = str(self.path)
        self.train_validation, self.test = torch.load(path)
        self.split_train_validation()
