from . import (
    arbitrary_correlated_features,
    base,
    bigger_net,
    bigger_net_random_inputs,
    correlated_features,
    iterative_sampling,
    random_inputs,
)


def get_algorithms():
    return (
        arbitrary_correlated_features,
        base,
        bigger_net,
        bigger_net_random_inputs,
        correlated_features,
        *iterative_sampling.get_algorithms(),
        random_inputs,
    )
