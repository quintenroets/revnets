import math

import torch
import torch.nn.functional as F
from torch.nn import init

from revnets.evaluations.weights.standardize.standardize import get_layers

from . import learning_rate_scheduler


class ReconstructModel(learning_rate_scheduler.ReconstructModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = 10

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if len(self.losses) > self.window_size:
            window = self.losses[-self.window_size :]
            local_optimum = window[0] == min(window)
            if local_optimum:
                self.check_dead_neurons()

    def check_dead_neurons(self) -> None:
        layers = get_layers(self.model)
        for layer in layers:
            self.check_layer_dead_neurons(layer)

    @classmethod
    def check_layer_dead_neurons(cls, layer):
        outputs = cls.get_output_activations(layer, n_check=2)
        dead_indices = torch.where(outputs < 1e-6)[0]
        for n_check in (5, 10, 30, 100, 300):
            if dead_indices.numel() > 0:
                outputs = cls.get_output_activations(layer, n_check=n_check)
                dead_indices = torch.where(outputs < 1e-6)[0]

        if dead_indices.numel() > 0:
            message_indices = dead_indices.cpu().numpy()
            message_indices = ", ".join(str(index) for index in message_indices)
            message = f"Repairing dead neurons: {message_indices}"
            print(message)
            cls.revive_dead_neurons(layer, dead_indices)

    @classmethod
    def get_output_activations(cls, layer, n_check=100):
        weight = layer.weight
        input_shape = n_check, weight.shape[1]
        inputs = torch.randn(input_shape, dtype=weight.dtype)
        inputs = inputs.to(layer.weight.device)
        with torch.no_grad():
            outputs = layer(inputs)
        outputs = F.relu(outputs).sum(dim=0)
        return outputs

    @classmethod
    def revive_dead_neurons(cls, layer, indices):
        state_dict = layer.state_dict()
        cls.revive_state_dict(state_dict, indices)
        layer.load_state_dict(state_dict)

    @classmethod
    def revive_state_dict(cls, state_dict, indices):
        # reinitialize biases
        zeros = torch.zeros_like(indices, dtype=state_dict["bias"].dtype)
        state_dict["bias"][indices] = zeros

        # reinitialize weights
        dead_weights = state_dict["weight"][indices]
        reinitialized_weights = torch.empty_like(dead_weights)
        init.kaiming_uniform_(reinitialized_weights, a=math.sqrt(5))
        state_dict["weight"][indices] = reinitialized_weights
