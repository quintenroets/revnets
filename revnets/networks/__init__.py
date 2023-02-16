from . import mediumnet, mediumnet_untrained, mininet, mininet_untrained


def get_all_networks():
    return mininet, mininet_untrained, mediumnet, mediumnet_untrained
