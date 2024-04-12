from dataclasses import dataclass

import torch

from . import correlated_features


@dataclass
class Reconstructor(correlated_features.Reconstructor):
    covariance_scale: float = 0.1

    def infer_distribution_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        train_inputs = self.extract_flattened_train_inputs()
        n_features = train_inputs.shape[1]
        means = torch.zeros(n_features)
        non_diag = 1 - torch.eye(n_features)
        covariances = torch.eye(n_features) + non_diag / n_features
        return means, covariances
