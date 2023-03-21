import json

from . import layers_mae


class Evaluator(layers_mae.Evaluator):
    def calculate_distance(self):
        return {
            name: self.calculate_weights_distance(original, reconstructed)
            for name, original, reconstructed in self.iterate_named_compared_layers()
        }

    def iterate_named_compared_layers(self):
        keys = self.original.state_dict().keys()
        original_values = self.original.state_dict().values()
        reconstruction_values = self.reconstruction.state_dict().values()
        yield from zip(keys, original_values, reconstruction_values)

    @classmethod
    def format_evaluation(cls, value, precision=3) -> str:
        if value:
            values = {
                name: super(layers_mae.Evaluator, cls).format_evaluation(layer_value)
                for name, layer_value in value.items()
            }
            formatted_value = json.dumps(values, indent=4)
        else:
            formatted_value = "/"
        return formatted_value
