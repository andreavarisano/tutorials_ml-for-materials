"""ROUGH DRAFT: create PyTorch datasets and dataloaders for regression.

Expected inputs
---------------
Features and targets should already have been split and preprocessed (scaled)ù
Keep the same material-row order used throughout the feature-construction exercises.

Suggested tensor shapes:

    features: (number_of_samples, number_of_features)
    targets:  (number_of_samples, 1)

Optional: convert to float32 tensors from float64 NumPy arrays to save memory and speed up training.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ArrayRegressionDataset(Dataset):
    """Wrap aligned NumPy feature and target arrays as a PyTorch Dataset."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Validate, convert, and store one regression partition.

        TODO
        ----
        - Require finite, non-empty NumPy arrays.
        - Require a two-dimensional feature matrix.
        - Accept targets shaped (n_samples,) or (n_samples, 1).
        - Convert one-dimensional targets to a single column.
        - Check that features and targets contain the same number of rows.
        - Convert both arrays to torch.float32 tensors.
        - Avoid unnecessary copies where it is safe to do so.
        """
        raise NotImplementedError(
            "Validate arrays and create feature and target tensors"
        )

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        raise NotImplementedError("Return the number of dataset rows")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one aligned feature-target pair."""
        raise NotImplementedError("Return tensors for one sample")


@dataclass
class RegressionDataLoaders:
    """Container for train, optional validation, and test dataloaders."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader


def create_regression_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 64,
    random_seed: int = 42,
    num_workers: int = 0,
    ) -> RegressionDataLoaders:
    """Create deterministic dataloaders from positional partition indices.

    TODO
    ----
    - Validate batch_size, indices, and partition non-overlap.
    - Slice features and targets using the supplied positional indices.
    - Create one ArrayRegressionDataset per non-empty partition.
    - Shuffle training samples only.
    - Pass a seeded torch.Generator to the training DataLoader.
    - Do not shuffle validation or test samples.
    - Keep drop_last=False for regression metrics over the complete dataset.
    - Return validation=None when validation_indices is empty.

    Beware
    ------
    The arrays supplied here must already have been transformed with scalers fitted on training data.
    """
    raise NotImplementedError(
        "Create aligned train, validation, and test dataloaders"
    )


def main() -> None:
    """Run a small shape and batching smoke test."""
    raise NotImplementedError(
        "Load prepared arrays and inspect a few dataloader batches"
    )


if __name__ == "__main__":
    main()
