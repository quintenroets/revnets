import pickle

import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset

from ..utils import Path
from . import base
from .utils import split_train_val


class Dataset(base.Dataset):
    def __init__(self):
        super().__init__()
        self.path = Path.data / "mnist_1D.pkl"

    def setup(self, stage: str = None) -> None:
        data = self.get_data()

        x = data["x"]
        y = data["y"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        def prepare_x(x_array: np.ndarray):
            x_torch = torch.Tensor(x_array)
            x_torch = torch.nn.functional.normalize(x_torch)
            return x_torch

        x = prepare_x(x)
        x_test = prepare_x(x_test)

        y = torch.LongTensor(y)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(x, y)
        self.train_dataset, self.val_dataset = split_train_val(train_dataset)
        self.test_dataset = TensorDataset(x_test, y_test)

    def get_data(self):
        self.prepare_data()
        with self.path.open("rb") as fp:
            return pickle.load(fp)

    def prepare_data(self):
        if not self.path.exists():
            url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
            self.path.byte_content = requests.get(url, allow_redirects=True).content
