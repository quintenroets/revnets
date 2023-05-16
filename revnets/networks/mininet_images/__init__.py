from . import mininet_100, mininet_128, mininet_200, mininet_small


def get_all_networks():
    return mininet_small, mininet_100, mininet_128, mininet_200


def get_networks():
    return mininet_small
