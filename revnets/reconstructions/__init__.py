from . import (
    cheat,
    empty,
    retrain,
    retrain_on_outputs,
    retrain_on_outputs_with_bigger_net,
)


def get_algorithms():
    return retrain, retrain_on_outputs, retrain_on_outputs_with_bigger_net
