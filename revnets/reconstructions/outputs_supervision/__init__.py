from . import (
    base,
    bigger_net,
    bigger_net_random_inputs,
    correlated_features,
    iterative_sampling,
    random_inputs,
)


def get_algorithms():
    return (
        base,
        bigger_net,
        bigger_net_random_inputs,
        correlated_features,
        *iterative_sampling.get_algorithms(),
        random_inputs,
    )
