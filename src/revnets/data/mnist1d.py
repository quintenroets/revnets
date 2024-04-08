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


@dataclass
class Dataset(base.Dataset):
    path: Path = Path.data / "mnist_1D.pkl"

    def prepare_data(self) -> None:
        data = self.load_data()

        scaler = StandardScaler()
        data.x = scaler.fit_transform(data.x)
        data.x_test = scaler.transform(data.x_test)

        x = torch.Tensor(data.x)
        x_test = torch.Tensor(data.x_test)

        y = torch.LongTensor(data.y)
        y_test = torch.LongTensor(data.y_test)

        self.train_val_dataset = TensorDataset(x, y)
        self.test_dataset = TensorDataset(x_test, y_test)

    def load_data(self) -> RawData:
        self.check_download()
        with self.path.open("rb") as fp:
            data = pickle.load(fp)
        return RawData(data["x"], data["y"], data["x_test"], data["y_test"])

    def check_download(self) -> None:
        if not self.path.exists():
            url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
            self.path.byte_content = requests.get(url, allow_redirects=True).content
