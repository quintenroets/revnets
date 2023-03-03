from . import val


class Evaluator(val.Evaluator):
    @classmethod
    def get_dataloader(cls, dataset):
        return dataset.train_dataloader(shuffle=False)
