import torch

from . import outputs, weights

evaluators = (weights, outputs)


def evaluate(*args, **kwargs):
    for evaluator in evaluators:
        evaluator.evaluate(*args, **kwargs)
