from dataclasses import dataclass

import torch

from . import correlated_features


@dataclass
class Dataset(correlated_features.Dataset):
    covariance_scale: float = 0.1

    def get_distribution_parameters(self):
        train_inputs = self.get_train_inputs().numpy()
        train_inputs = train_inputs.reshape(len(train_inputs), -1)
        n_features = train_inputs.shape[1]
        means = torch.zeros(n_features)
        non_diag = 1 - torch.eye(n_features)
        covariances = torch.eye(n_features) + non_diag / n_features
        return means, covariances
