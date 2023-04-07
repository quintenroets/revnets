from . import layers_mae


class Evaluator(layers_mae.Evaluator):
    def evaluate(self):
        raise NotImplementedError
