import torch

from revnets.networks import models

from ...utils import config
from . import random_inputs


class Reconstructor(random_inputs.Reconstructor):
    @classmethod
    def initialize_reconstruction(cls):
        torch.manual_seed(config.manual_seed * 2)
        return models.mediumnet.Model(hidden_size1=200, hidden_size2=100)
