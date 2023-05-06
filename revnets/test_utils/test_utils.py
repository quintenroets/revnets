import torch
from hypothesis import strategies

from revnets.networks import models
from revnets.utils import config

MAX_SIZE = 100


def get_input_size():
    model = initialize_model()
    return model.layer1.weight.shape[1]


def network_inputs():
    floats = strategies.floats(min_value=-10.0, max_value=10.0)
    input_size = get_input_size()
    size = min(input_size, MAX_SIZE)
    list_of_floats = strategies.lists(elements=floats, min_size=size, max_size=size)
    return list_of_floats


def initialize_model():
    torch.manual_seed(config.manual_seed)
    return models.mediumnet_images.Model(
        input_size=784, hidden_size1=100, hidden_size2=50
    )
    # return models.mediumnet.Model()


def prepare_inputs(inputs: list[float]):
    deterministic_size = get_input_size() - MAX_SIZE
    inputs = inputs + [0] * deterministic_size
    return torch.Tensor(inputs).unsqueeze(0)
