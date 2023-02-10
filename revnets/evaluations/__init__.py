import torch

from . import weights

evaluators = (weights,)


def evaluate(original: torch.nn.Module, reconstruction: torch.nn.Module):
    for evaluator in evaluators:
        evaluator.evaluate(original, reconstruction)
