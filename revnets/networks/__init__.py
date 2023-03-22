from . import mediumnet, mediumnet_untrained, mininet, mininet_untrained
from .base import Network


def get_all_networks():
    return (mininet,)
    return mininet, mininet_untrained, mediumnet, mediumnet_untrained


def get_networks():
    return (mininet,)
