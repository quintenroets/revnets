from . import mediumnet, mediumnet_images, mininet
from .base import Network


def get_all_networks():
    return (
        *mininet.get_all_networks(),
        *mediumnet.get_all_networks(),
        *mediumnet_images.get_networks(),
    )


def get_networks():
    return (
        *mininet.get_networks(),
        *mediumnet.get_networks(),
        *mediumnet_images.get_networks(),
    )
