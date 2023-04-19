from dataclasses import dataclass

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from revnets.data import output_supervision
from revnets.utils.colors import get_colors

from . import difficult_train_inputs


@dataclass
class Reconstructor(difficult_train_inputs.Reconstructor):
    n_rounds: int = 5

    def __post_init__(self):
        super().__post_init__()
        self.n_rounds = 1

    def run_round(self):
        self.train_model(self.data)
        self.analyze_samplings()
        self.add_difficult_samples()

    def analyze_samplings(self):
        train_inputs = self.data.get_all_inputs(output_supervision.Split.train)
        train_losses = self.get_sorted_losses(train_inputs)
        elbow = self.get_elbow(train_losses)
        elbow_y = train_losses[elbow]

        difficult_existing_inputs = self.extract_difficult_inputs()
        self.visualize_samples(difficult_existing_inputs)
        difficult_inputs = self.sample_difficult_inputs()
        self.visualize_samples(difficult_inputs)
        random_inputs = self.data.generate_random_inputs(difficult_inputs.shape)
        random_scales = [0, 1, 1 / 10, 1 / 100]

        recombined_inputs = [
            difficult_inputs + random_inputs * scale for scale in random_scales
        ]
        analyzed_inputs = [
            train_inputs,
            random_inputs,
            2 * random_inputs,
            *recombined_inputs,
        ]

        labels = [
            "train inputs",
            "random inputs",
            "large random inputs",
            "recombined difficult samples",
            "recombined difficult samples + noise",
            "recombined difficult samples + noise (scale 1/10)",
            "recombined difficult samples + noise (scale 1/100)",
        ]

        colors = get_colors()
        for inputs, color, label in zip(analyzed_inputs, colors, labels):
            losses = self.get_sorted_losses(inputs) / inputs.std()
            plt.plot(losses, color=color, label=label)

        threshold_color = colors[len(analyzed_inputs)]
        plt.axvline(x=elbow, label="High loss threshold", color=threshold_color)
        plt.axhline(y=elbow_y, color=threshold_color)

        x_ticks = plt.xticks()
        new_x_ticks = self.get_new_ticks(x_ticks, elbow)
        plt.xticks(new_x_ticks)

        y_ticks = plt.yticks()
        new_y_ticks = self.get_new_ticks(y_ticks, elbow_y)
        plt.yticks(new_y_ticks)

        plt.xlabel("Input index")
        plt.ylabel("Reconstruction loss")
        plt.title("Sorted losses")
        plt.legend()
        plt.show()

    @classmethod
    def get_new_ticks(cls, ticks, extra_tick):
        ticks = [t for t in list(ticks)[0] if t >= 0]
        return [*ticks, extra_tick]

    def get_sorted_losses(self, inputs):
        dataset = TensorDataset(inputs)
        outputs = self.get_predictions(dataset, self.reconstruction)
        targets = self.get_predictions(dataset, self.original)
        return self.calculate_sorted_losses(outputs, targets)[0]

    def get_predictions(self, dataset, model):
        dataloader = DataLoader(dataset, batch_size=self.data.eval_batch_size)
        return self.data.get_predictions(dataloader, model)

    @classmethod
    def visualize_samples(cls, samples):
        colors = get_colors()
        fig, ax = plt.subplots(figsize=(10, 10))
        for features in samples:
            ax.plot(features, color=colors[0], alpha=0.3, linewidth=0.1)

        plt.title("Parallel Coordinates Plot")
        plt.xlabel("Feature")
        plt.ylabel("Feature Value")
        plt.show()
