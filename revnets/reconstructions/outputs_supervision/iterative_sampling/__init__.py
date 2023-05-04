from . import analysis, base, difficult_inputs, difficult_train_inputs, balanced_outputs


def get_algorithms():
    return (
        difficult_inputs,
        difficult_train_inputs,
        analysis,
    )
