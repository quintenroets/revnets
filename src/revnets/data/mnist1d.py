import pickle

import numpy as np
import requests
import torch
from numpy.typing import NDArray
from package_utils.dataclasses import SerializationMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ..models import Path
from . import base


class RawData(SerializationMixin):
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.float32]


class Dataset(base.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = Path.data / "mnist_1D.pkl"

    def prepare_data(self) -> None:
        data = self.get_data()

        scaler = StandardScaler()
        data.x = scaler.fit_transform(data.x)
        data.x_test = scaler.transform(data.x_test)

        x = torch.Tensor(data.x)
        x_test = torch.Tensor(data.x_test)

        y = torch.LongTensor(data.y)
        y_test = torch.LongTensor(data.y_test)

        self.train_val_dataset = TensorDataset(x, y)
        self.test_dataset = TensorDataset(x_test, y_test)

    def get_data(self) -> RawData:
        self.check_download()
        with self.path.open("rb") as fp:
            data = pickle.load(fp)
        return RawData.from_dict(data)

    def check_download(self) -> None:
        if not self.path.exists():
            url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
            self.path.byte_content = requests.get(url, allow_redirects=True).content
