from . import analysis, balanced_outputs, base, difficult_inputs, difficult_train_inputs


def get_algorithms():
    return (
        difficult_inputs,
        difficult_train_inputs,
        analysis,
    )
