from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from . import random


@dataclass
class Dataset(random.Dataset):
    def get_distribution_parameters(self):
        train_inputs = self.get_train_inputs().numpy()

        train_inputs = train_inputs.reshape(len(train_inputs), -1)
        train_means = train_inputs.mean(axis=0)
        train_inputs -= train_means

        dtype = self.get_dtype()
        covariance_matrix = np.cov(train_inputs, rowvar=False)
        covariance_matrix = torch.tensor(covariance_matrix, dtype=dtype)
        covariance_matrix = self.regularize_covariance_matrix(covariance_matrix)

        means = torch.tensor(train_means, dtype=dtype)
        return means, covariance_matrix

    def generate_random_inputs(self, shape):
        means, covariance_matrix = self.get_distribution_parameters()
        distribution = MultivariateNormal(means, covariance_matrix)
        sample_shape = (shape[0],)
        # same mean, variance, and covariance as the training data
        samples = distribution.sample(sample_shape)  # noqa
        return samples

    def regularize_covariance_matrix(self, matrix, eps: float = 10e-3):
        dtype = self.get_dtype()
        shape = matrix.shape[0]
        regularization = torch.eye(shape, dtype=dtype) * eps
        return matrix + regularization
