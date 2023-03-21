import numpy as np
import torch
from plib import Path

from ..evaluations.weights import mae, named_layers_mae
from ..evaluations.weights.standardize import standardize
from ..networks import mininet
from ..reconstructions.outputs_supervision import random_inputs


class Experiment:
    @classmethod
    def run(cls):
        with torch.no_grad():
            cls.analyze()

    @classmethod
    def analyze(cls):
        network = mininet.Network()

        model = network.get_trained_network()
        reconstructor = random_inputs.Reconstructor(model, network)
        reconstruction = reconstructor.reconstruct()
        # pprint(reconstruction.state_dict())

        standardize.standardize(reconstruction)
        evaluator = mae.Evaluator(model, reconstruction, network)
        evaluator2 = named_layers_mae.Evaluator(model, reconstruction, network)

        for ev in (evaluator, evaluator2):
            distance = ev.evaluate()
            pprint(distance)

        total_weight = 0
        total_size = 0
        for _, layer in evaluator.iterate_compared_layers():
            total_weight += torch.abs(layer).sum()
            total_size += layer.numel()

        pprint(total_weight / total_size)

        return
        for name, layer1, layer2 in evaluator2.iterate_named_compared_layers():
            layer = layer1 - layer2
            absdif = torch.abs(layer)
            maxas = absdif.max()
            minas = absdif.min()
            pprint(maxas, minas)
            continue
            torch.save(layer, Path.docs / "out")
            arr = layer.detach().numpy()
            np.savetxt("out2.txt", arr)
