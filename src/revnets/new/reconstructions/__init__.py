from . import cheat, empty, outputs_supervision, retrain


def get_algorithms():
    return retrain, *outputs_supervision.get_algorithms()
