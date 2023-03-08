from . import base, bigger_net, bigger_net_random_inputs, random_inputs


def get_algorithms():
    return base, bigger_net_random_inputs
    return base, random_inputs, bigger_net_random_inputs
