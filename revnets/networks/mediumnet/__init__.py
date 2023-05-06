from . import mediumnet, mediumnet_20, mediumnet_40, mediumnet_untrained


def get_all_networks():
    return (
        mediumnet,
        mediumnet_20,
        mediumnet_40,
        mediumnet_untrained,
    )


def get_networks():
    return mediumnet_20, mediumnet_40
