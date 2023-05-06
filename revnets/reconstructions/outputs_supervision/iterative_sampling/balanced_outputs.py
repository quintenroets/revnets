from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from . import difficult_inputs


class InputModel(pl.LightningModule):
    def __init__(
        self, model, feature_size, desired_output, learning_rate=0.01, verbose=True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.desired_output = torch.nn.Embedding.from_pretrained(desired_output)
        shape = desired_output.shape[0], feature_size
        self.inputs_embedding = self.get_input_embeddings(shape)
        self.model = model
        self.verbose = verbose

    def on_train_start(self) -> None:
        if self.verbose:
            print("\nAverage difference: ", end="\n\t")

    @classmethod
    def get_input_embeddings(cls, shape) -> torch.nn.Embedding:
        embeddings = torch.nn.Embedding(*shape)
        torch.nn.init.normal_(embeddings.weight)
        return embeddings

    def forward(self, _):
        output = self.model(self.inputs_embedding.weight)
        self.model.zero_grad()
        return output

    def calculate_loss(self, outputs):
        difference = outputs - self.desired_output.weight
        self.desired_output.zero_grad()
        loss = difference.norm()
        if self.verbose:
            print(f"{loss.item():.3f}", end=" ")
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        return self.calculate_loss(outputs)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.inputs_embedding.parameters(), lr=self.learning_rate
        )

    def get_optimized_inputs(self):
        return self.inputs_embedding.weight.detach()


@dataclass
class Reconstructor(difficult_inputs.Reconstructor):
    def prepare_train_data_inputs(self, data):
        self.data = data
        output_size = 10
        desired_output = torch.eye(output_size)
        ones = torch.ones((self.num_samples, 1))
        zeros = torch.zeros((self.num_samples, output_size - 1))
        desired_output = torch.cat((ones, zeros), dim=1)
        model = InputModel(self.original, self.feature_size, desired_output)
        self.fit_inputs_model(model, epochs=1000)
        inputs = model.get_optimized_inputs()
        with torch.no_grad():
            outputs = self.original(inputs)
        difference = outputs - desired_output
        differences = difference.sum(axis=1)
        min_index = differences.argmin()
        best_output = outputs[min_index]
        plt.plot(best_output)
        # for i, out in enumerate(outputs):
        # plt.plot(out, label=i)
        plt.show()

        exit()
