from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


LOGGER = logging.getLogger(__name__)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class FitnessModel(Protocol):
    def train(self, dataset) -> None: ...

    def get_fitness(self, sequences) -> torch.Tensor: ...


def normalize_sequence(sequence: Any) -> str:
    if isinstance(sequence, str):
        return sequence
    if isinstance(sequence, np.ndarray):
        sequence = sequence.tolist()
    return "".join(str(token) for token in sequence)


def normalize_sequences(sequences) -> list[str]:
    return [normalize_sequence(sequence) for sequence in sequences]


def sequences_to_tensor(sequences, alphabet: str = AMINO_ACIDS) -> torch.Tensor:
    token_by_amino_acid = {
        amino_acid: index for index, amino_acid in enumerate(alphabet)
    }
    encoded = [
        functional.one_hot(
            torch.tensor([token_by_amino_acid[amino_acid] for amino_acid in sequence]),
            num_classes=len(alphabet),
        )
        for sequence in sequences
    ]
    return torch.stack(encoded).permute(0, 2, 1).float()


class ConvolutionalNetwork(nn.Module):
    """CNN surrogate used in the original ProSpero experiments."""

    def __init__(
        self,
        sequence_length: int,
        input_channels: int = 20,
        filters: int = 32,
        hidden_dimension: int = 128,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv1d(
            input_channels, filters, kernel_size, padding="valid"
        )
        self.convolution_2 = nn.Conv1d(filters, filters, kernel_size, padding="same")
        self.convolution_3 = nn.Conv1d(filters, filters, kernel_size, padding="same")
        self.global_max_pool = nn.MaxPool1d(kernel_size=sequence_length - 4)
        self.hidden_1 = nn.Linear(filters, hidden_dimension)
        self.hidden_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(hidden_dimension, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.convolution_1(inputs))
        activations = torch.relu(self.convolution_2(activations))
        activations = torch.relu(self.convolution_3(activations))
        activations = self.global_max_pool(activations).squeeze(dim=-1)
        activations = torch.relu(self.hidden_1(activations))
        activations = torch.relu(self.hidden_2(activations))
        return self.output(self.dropout(activations))


class ConvolutionalSurrogate:
    def __init__(self, sequence_length: int, config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = ConvolutionalNetwork(sequence_length).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss = nn.MSELoss()

    def data_loader(self, sequences, labels, *, shuffle: bool):
        dataset = torch.utils.data.TensorDataset(
            sequences_to_tensor(sequences), torch.as_tensor(labels).float()
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.surrogate_batch_size,
            shuffle=shuffle,
        )

    def batch_loss(self, batch) -> torch.Tensor:
        sequences, labels = batch
        predictions = self.network(sequences.to(self.device)).squeeze(dim=-1)
        return self.loss(predictions, labels.to(self.device))

    def train(self, dataset) -> None:
        training = self.data_loader(dataset.train, dataset.train_scores, shuffle=True)
        validation = self.data_loader(
            dataset.valid, dataset.valid_scores, shuffle=False
        )
        best_validation_loss = np.inf
        stale_epochs = 0
        for epoch in range(self.config.maximum_epochs):
            self.network.train()
            for batch in training:
                loss = self.batch_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % self.config.validation_frequency:
                continue
            self.network.eval()
            with torch.no_grad():
                validation_loss = np.mean(
                    [self.batch_loss(batch).item() for batch in validation]
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break

    def get_fitness(self, sequences) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            inputs = sequences_to_tensor(sequences).to(self.device)
            return self.network(inputs).squeeze(dim=-1)


class Ensemble:
    def __init__(self, models: list[FitnessModel]) -> None:
        if not models:
            raise ValueError("A surrogate ensemble cannot be empty.")
        self.models = models

    def train(self, dataset) -> None:
        LOGGER.info("Training surrogate ensemble on %d samples", len(dataset.train))
        for model in self.models:
            model.train(dataset)

    @torch.no_grad()
    def get_scores(self, sequences) -> torch.Tensor:
        return self._predictions(sequences).mean(dim=0)

    @torch.no_grad()
    def forward_with_uncertainty(self, sequences) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = self._predictions(sequences)
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)

    @torch.no_grad()
    def get_ucb(self, sequences, k: float = 0.1) -> torch.Tensor:
        mean, standard_deviation = self.forward_with_uncertainty(sequences)
        return mean + k * standard_deviation

    def _predictions(self, sequences) -> torch.Tensor:
        return torch.stack([model.get_fitness(sequences) for model in self.models])


def build_surrogate_model(sequence_length: int, config) -> ConvolutionalSurrogate:
    if config.surrogate_architecture != "cnn":
        raise ValueError(
            "The paper reproduction supports only the published CNN surrogate."
        )
    return ConvolutionalSurrogate(sequence_length, config)
