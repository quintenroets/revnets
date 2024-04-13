from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import TensorDataset

from revnets.context import context
from revnets.data import base
from revnets.pipelines import Pipeline
from revnets.utils.data import compute_targets


class QueryDataSet(TensorDataset):
    def __init__(
        self, target: nn.Module, evaluation_batch_size: int | None = None
    ) -> None:
        self.target = target
        if evaluation_batch_size is None:
            evaluation_batch_size = context.config.evaluation_batch_size
        self.evaluation_batch_size = evaluation_batch_size
        tensors = torch.Tensor([]), torch.Tensor([])
        super().__init__(*tensors)

    def add(self, queries: torch.Tensor) -> None:
        targets = compute_targets(queries, self.target, self.evaluation_batch_size)
        self.tensors = (
            torch.cat([self.tensors[0], queries], dim=0),
            torch.cat([self.tensors[1], targets], dim=0),
        )


@dataclass
class DataModule(base.DataModule):
    pipeline: Pipeline = field(default_factory=Pipeline)
    batch_size: int = field(
        default_factory=lambda: context.config.reconstruction_training.batch_size
    )
    evaluation_batch_size: int = field(
        default_factory=lambda: context.config.evaluation_batch_size
    )
    train: QueryDataSet = field(init=False)
    validation: QueryDataSet = field(init=False)
    test: QueryDataSet = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        target = self.pipeline.target
        self.train = QueryDataSet(target, self.evaluation_batch_size)
        self.validation = QueryDataSet(target, self.evaluation_batch_size)
        self.test = QueryDataSet(target, self.evaluation_batch_size)
