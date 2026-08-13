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
        # input validation
        if not isinstance(features, np.ndarray):
            raise TypeError(f"features must be a NumPy array, not {type(features)}")
        if not isinstance(targets, np.ndarray):
            raise TypeError(f"targets must be a NumPy array, not {type(targets)}")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        if not np.isfinite(targets).all():
            raise ValueError("targets must contain only finite values")
        if features.size == 0:
            raise ValueError("features cannot be empty")
        if targets.size == 0:
            raise ValueError("targets cannot be empty")
        if features.ndim != 2:
            raise ValueError(f"features must be two-dimensional, not {features.ndim}")
        
        if targets.ndim == 1:
            targets_2d = targets.reshape(-1, 1)
        elif targets.ndim == 2:
          if targets.shape[1] > 1:
            raise ValueError("targets must be a single column.")
          targets_2d = targets
        else:
          raise ValueError("targets must be a one-dimensional or two-dimensional array.")
        
        if features.shape[0] != targets_2d.shape[0]:
            raise ValueError("features and targets must have the same number of rows")
        
        # conversion to torch.float32 tensors
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets_2d, dtype=torch.float32)


    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one aligned feature-target pair."""
        return self.features[index], self.targets[index]

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
    # input validation
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer")
    if train_indices.size == 0:
        raise ValueError("train_indices cannot be empty")
    if test_indices.size == 0:
        raise ValueError("test_indices cannot be empty")
    if np.intersect1d(train_indices, validation_indices).size > 0:
        raise ValueError("Train and validation indices overlap.")
    if np.intersect1d(train_indices, test_indices).size > 0:
        raise ValueError("Train and test indices overlap.")
    if np.intersect1d(validation_indices, test_indices).size > 0:
        raise ValueError("Validation and test indices overlap.")
    
    # slice features and targets to create datasets and dataloaders
    ## train
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    train_dataset = ArrayRegressionDataset(train_features, train_targets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(random_seed),
        drop_last=False,
    )
    ## validation
    if validation_indices.size > 0:
        validation_features = features[validation_indices]
        validation_targets = targets[validation_indices]
        validation_dataset = ArrayRegressionDataset(validation_features, validation_targets)
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        validation_dataloader = None
    ## test
    test_features = features[test_indices]
    test_targets = targets[test_indices]
    test_dataset = ArrayRegressionDataset(test_features, test_targets)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return RegressionDataLoaders(
        train=train_dataloader,
        validation=validation_dataloader,
        test=test_dataloader,
    )

def main() -> None:
    """Run a small shape and batching smoke test."""
    print("Smoke Test")
    
    # dataset of 100 samples and 5 features
    n_samples = 100
    n_features = 5
    
    features = np.random.randn(n_samples, n_features)
    targets = np.random.randn(n_samples)  # 1D targets
    
    # split indices
    train_idx = np.arange(0, 70)       # 70 samples
    val_idx = np.arange(70, 85)        # 15 samples
    test_idx = np.arange(85, 100)      # 15 samples
    
    batch_size = 10
    
    print(f"Creating DataLoaders...")
    loaders = create_regression_dataloaders(
        features=features,
        targets=targets,
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
        batch_size=batch_size
    )
    
    print("\nInspecting Train DataLoader")
    # Iterate through one epoch of the training loader
    for batch_idx, (X_batch, y_batch) in enumerate(loaders.train):
        print(f"Batch {batch_idx + 1}: X shape {X_batch.shape}, y shape {y_batch.shape}")
        
    print("\nInspecting Validation DataLoader")
    for batch_idx, (X_batch, y_batch) in enumerate(loaders.validation):
        print(f"Batch {batch_idx + 1}: X shape {X_batch.shape}, y shape {y_batch.shape}")
    
    print("\nInspecting Test DataLoader")
    for batch_idx, (X_batch, y_batch) in enumerate(loaders.test):
        print(f"Batch {batch_idx + 1}: X shape {X_batch.shape}, y shape {y_batch.shape}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
