from . import mininet, mininet_100, mininet_bigger_reconstruction, mininet_untrained


def get_all_networks():
    return mininet, mininet_100, mininet_untrained


def get_networks():
    return (mininet_100,)
