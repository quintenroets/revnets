from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.distributions.multivariate_normal import MultivariateNormal

from revnets.context import context

from . import random


@dataclass
class Reconstructor(random.Reconstructor):
    epsilon: float = 1e-3

    def create_random_inputs(self, shape: Sequence[int]) -> torch.Tensor:
        means, covariance_matrix = self.infer_distribution_parameters()
        distribution = MultivariateNormal(means, covariance_matrix)  # type: ignore[no-untyped-call]
        sample_shape = torch.Size((shape[0],))
        # same mean, variance, and covariance as the training data
        return distribution.sample(sample_shape).reshape(shape)

    def infer_distribution_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        train_inputs = self.extract_flattened_train_inputs()
        train_means = train_inputs.mean(axis=0)
        train_inputs -= train_means

        covariance_matrix_numpy = np.cov(train_inputs, rowvar=False)
        covariance_matrix = torch.tensor(covariance_matrix_numpy, dtype=context.dtype)
        covariance_matrix = self.regularize_covariance_matrix(covariance_matrix)

        means = torch.tensor(train_means, dtype=context.dtype)
        return means, covariance_matrix

    def extract_flattened_train_inputs(self) -> NDArray[np.float64]:
        inputs = self.pipeline.load_all_train_inputs().numpy()
        flattened_inputs = inputs.reshape(len(inputs), -1)
        return cast("NDArray[np.float64]", flattened_inputs)

    def regularize_covariance_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        shape = matrix.shape[0]
        regularization = torch.eye(shape, dtype=context.dtype) * self.epsilon
        return matrix + regularization
