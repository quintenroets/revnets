from . import (
    base,
    bigger_net,
    bigger_net_random_inputs,
    iterative_sampling,
    random_inputs,
    smaller_weight_variance,
)


def get_algorithms():
    return (
        base,
        bigger_net,
        bigger_net_random_inputs,
        *iterative_sampling.get_algorithms(),
        random_inputs,
        smaller_weight_variance,
    )
