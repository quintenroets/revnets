import pickle

import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ..utils import Path
from . import base


class Dataset(base.Dataset):
    def __init__(self):
        super().__init__()
        self.path = Path.data / "mnist_1D.pkl"

    def prepare_data(self):
        data = self.get_data()

        x = data["x"]
        y = data["y"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x_test = scaler.transform(x_test)

        x = torch.Tensor(x)
        x_test = torch.Tensor(x_test)

        y = torch.LongTensor(y)
        y_test = torch.LongTensor(y_test)

        self.train_val_dataset = TensorDataset(x, y)
        self.test_dataset = TensorDataset(x_test, y_test)

    def get_data(self):
        self.check_download()
        with self.path.open("rb") as fp:
            return pickle.load(fp)

    def check_download(self):
        if not self.path.exists():
            url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
            self.path.byte_content = requests.get(url, allow_redirects=True).content
