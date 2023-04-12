from . import (
    base,
    bigger_net,
    bigger_net_random_inputs,
    iterative_sampling,
    iterative_sampling_analysis,
    random_inputs,
)


def get_algorithms():
    return (
        base,
        bigger_net,
        bigger_net_random_inputs,
        iterative_sampling,
        iterative_sampling_analysis,
        random_inputs,
    )
