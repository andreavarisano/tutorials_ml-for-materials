"""ROUGH DRAFT: train a small PyTorch MLP for scalar regression.

The model consumes one fixed-size feature vector per material and predicts one
scalar target. Start by deliberately overfitting a tiny subset. Only then train
on the complete training partition.

A tiny bit of knowledge: Lightning-inspired organization
---------------------------------------------------------
This file uses plain torch.nn.Module; it does not require Lightning. However,
the model exposes methods inspired by lightning.LightningModule:

    forward
    training_step
    validation_step
    test_step
    configure_optimizers

The model therefore owns the definition of a prediction, a batch loss, and its
optimizer configuration. The external loops still show explicitly what
Lightning would normally automate: moving batches to a device, clearing
gradients, calling backward, updating parameters, disabling gradients during
evaluation, and aggregating losses over an epoch.

This organization makes a later transition to Lightning easier without hiding
basic PyTorch mechanics now.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import sys
from pathlib import Path

feature_construction = (
    Path(__file__).resolve().parent.parent / "01_Feature_Construction"
)
if str(feature_construction) not in sys.path:
    sys.path.append(str(feature_construction))

import step01_load_dataset as load
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp
import step03_concatenate_features as conc
import step04_preprocessing as prep
import STUB_create_datasets_and_dataloaders as dl

RegressionBatch = tuple[torch.Tensor, torch.Tensor]
EvaluationStage = Literal["validation", "test"]


@dataclass
class TrainingHistory:
    """Loss values collected during neural-network training."""

    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)


class RegressionMLP(nn.Module):
    """MLP organized with Lightning-inspired training and evaluation hooks."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Sequence[int] = (128, 64),
        dropout_probability: float = 0.0,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """Construct the network, loss function, and optimizer configuration.

        TODO
        ----
        - Call super().__init__().
        - Validate dimensions, dropout_probability, learning_rate, and
          weight_decay.
        - Build Linear + activation blocks for the hidden dimensions.
        - Optionally insert Dropout.
        - Finish with nn.Linear(last_hidden_dimension, 1).
        - Store the network in an nn.Sequential or ModuleList.
        - Store an MSE loss function.
        - Store learning_rate and weight_decay for configure_optimizers.

        Lightning comparison
        --------------------
        A real LightningModule would commonly call save_hyperparameters().
        Here, store the values explicitly so their role remains visible.
        """
        super().__init__()

        # input validation
        if not isinstance(input_dimension, int) or input_dimension < 0:
            raise ValueError("input_dimension must be a non-negative integer")
        if not isinstance(hidden_dimensions, Sequence) or not all(
            isinstance(dimension, int) and dimension > 0
            for dimension in hidden_dimensions
        ):
            raise ValueError(
                "hidden_dimensions must be a sequence of positive integers"
            )
        if (
            not isinstance(dropout_probability, float)
            or dropout_probability < 0.0
            or dropout_probability > 1.0
        ):
            raise ValueError("dropout_probability must be a float between 0.0 and 1.0")
        if not isinstance(learning_rate, float) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive float")
        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("weight_decay must be a non-negative float")

        # network

        layers = []  # list of layers

        current_dimension = input_dimension

        for dimension in hidden_dimensions:
            layers.append(nn.Linear(current_dimension, dimension))
            layers.append(nn.ReLU())
            if dropout_probability > 0.0:
                layers.append(nn.Dropout(dropout_probability))
            current_dimension = dimension

        layers.append(nn.Linear(current_dimension, 1))

        self.network = nn.Sequential(*layers)

        # loss function and optimizer

        self.loss_function = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (batch_size, input_dimension) to (batch_size, 1)."""
        return self.network(features)

    def _shared_step(
        self,
        batch: RegressionBatch,
        batch_index: int,
    ) -> torch.Tensor:
        """Calculate the scalar regression loss shared by all stages.

        TODO
        ----
        - Unpack features and targets from batch.
        - Validate their ranks and matching batch dimension.
        - Calculate predictions by calling self(features), not forward directly.
        - Check that prediction and target shapes agree.
        - Calculate and return a scalar MSE tensor.
        - Keep the tensor attached to the computation graph.

        The batch_index argument mirrors Lightning's step signature. It may not
        be needed initially, but can later help with logging or debugging.
        """
        features, targets = batch

        # features and targets validation
        if (
            features.ndim != 2
            or targets.ndim != 2
            or features.shape[0] != targets.shape[0]
        ):
            raise ValueError(
                "features and targets must be (batch_size, input_dimension) and (batch_size, 1) tensors"
            )

        predictions = self(features)

        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape")

        return self.loss_function(predictions, targets)

    def training_step(
        self,
        batch: RegressionBatch,
        batch_index: int,
    ) -> torch.Tensor:
        """Return the differentiable loss for one training batch.

        TODO
        ----
        - Delegate prediction and loss calculation to _shared_step.
        - Return the scalar loss tensor.
        - Do not call backward or optimizer.step here; the outer plain-PyTorch
          loop performs those operations.

        Lightning comparison
        --------------------
        Lightning normally performs backward and optimizer stepping using the
        loss returned by training_step.
        """
        return self._shared_step(batch, batch_index)

    def validation_step(
        self,
        batch: RegressionBatch,
        batch_index: int,
    ) -> torch.Tensor:
        """Return the loss for one validation batch.

        TODO
        ----
        - Delegate to _shared_step.
        - Return a scalar loss tensor.
        - Do not update parameters or optimizer state.
        """
        return self._shared_step(batch, batch_index)

    def test_step(
        self,
        batch: RegressionBatch,
        batch_index: int,
    ) -> torch.Tensor:
        """Return the loss for one test batch without changing model state.

        TODO
        ----
        Keep this separate from validation_step so later code can report test
        metrics without accidentally using test performance for model choices.
        """
        return self._shared_step(batch, batch_index)

    def configure_optimizers(self) -> Optimizer:
        """Create and return the optimizer associated with this model.

        TODO
        ----
        - Create torch.optim.Adam over self.parameters().
        - Use the learning rate and weight decay stored during initialization.
        - Return the optimizer.

        Lightning comparison
        --------------------
        The plural method name matches Lightning's configure_optimizers hook,
        which can also return schedulers or multiple optimizers.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


def train_one_epoch(
    model: RegressionMLP,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch using model.training_step.

    TODO
    ----
    - Set model.train().
    - Move every feature and target batch to device.
    - Clear gradients before each optimization step.
    - Call model.training_step(batch, batch_index).
    - Call loss.backward() and optimizer.step().
    - Accumulate loss weighted by batch size.
    - Return the average over samples, not the average over batches.

    This loop deliberately exposes operations that Lightning would automate.
    """
    model.train()

    total_loss = 0.0
    samples = 0

    for batch_index, (features, target) in enumerate(dataloader):
        features, target = features.to(device), target.to(device)
        batch = (features, target)
        batch_size = features.shape[0]

        optimizer.zero_grad()
        loss = model.training_step(batch, batch_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        samples += batch_size

    return total_loss / samples


def evaluate_one_epoch(
    model: RegressionMLP,
    dataloader: DataLoader,
    device: torch.device,
    stage: EvaluationStage = "validation",
) -> float:
    """Run validation_step or test_step and return average loss.

    TODO
    ----
    - Accept only "validation" or "test".
    - Set model.eval().
    - Use torch.no_grad().
    - Move each batch to device.
    - Call the stage-appropriate model hook.
    - Never call backward or update the optimizer.
    - Return a sample-weighted average loss.
    """
    # input validation
    if stage not in ["validation", "test"]:
        raise ValueError("stage must be 'validation' or 'test'")

    model.eval()

    total_loss = 0.0
    samples = 0

    with torch.no_grad():
        for batch_index, (features, target) in enumerate(dataloader):
            features, target = features.to(device), target.to(device)
            batch = (features, target)
            batch_size = features.shape[0]

            if stage == "validation":
                loss = model.validation_step(batch, batch_index)
            elif stage == "test":
                loss = model.test_step(batch, batch_index)

            total_loss += loss.item() * batch_size
            samples += batch_size

    return total_loss / samples


def fit_neural_network(
    model: RegressionMLP,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    number_of_epochs: int = 100,
    device: torch.device | None = None,
) -> TrainingHistory:
    """Fit the MLP using its Lightning-inspired hooks.

    TODO
    ----
    - Validate number_of_epochs.
    - Select CPU or CUDA when device is None.
    - Move the model to device.
    - Obtain the optimizer from model.configure_optimizers().
    - Call train_one_epoch for every epoch.
    - Call evaluate_one_epoch with stage="validation" when available.
    - Record losses in TrainingHistory.
    - Print occasional progress without printing every batch.

    Possible extensions
    -------------------
    - Add self.log-like metric collection.
    - Early stopping and best-model checkpointing.
    - Learning-rate scheduling from configure_optimizers.
    - MAE in original target units after inverse scaling.
    - Convert RegressionMLP into an actual LightningModule.
    """
    if not isinstance(number_of_epochs, int) or number_of_epochs <= 0:
        raise ValueError("number_of_epoch must be a non-zero integer")

    if device == None:
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

    model.to(device)

    optimizer = model.configure_optimizers()

    training_history = TrainingHistory()

    for epoch in range(number_of_epochs):
        # training
        train_loss = train_one_epoch(model, training_dataloader, optimizer, device)
        training_history.training_loss.append(train_loss)

        # validation
        val_loss = None
        if validation_dataloader is not None:
            val_loss = evaluate_one_epoch(
                model, validation_dataloader, device, "validation"
            )
            training_history.validation_loss.append(val_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{number_of_epochs}]")
            print(f"  Training loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Validation loss: {val_loss:.4f}")

    return training_history


def main() -> None:  # a better scaling solution should be implemented
    """Create the MLP, overfit a tiny subset, and start full training."""

    # training only on combined features

    print("=== Dataset and Preprocessing ===")

    # === Dataset ===
    print("Loading dataset...")
    dataframe = load.load_dielectric_dataset()

    print("Splitting dataset...")
    n_samples = len(dataframe)
    validation_fraction = 0.15
    test_fraction = 0.15
    random_seed = 42

    split = load.create_dataset_split(
        n_samples,
        validation_fraction,
        test_fraction,
        random_seed,
    )

    print(f"Number of samples: {n_samples}")
    print(f"Training fraction: {(1 - validation_fraction - test_fraction)}")
    print(f"Validation fraction: {validation_fraction}")
    print(f"Test fraction: {test_fraction}")
    print(f"Random seed: {random_seed}")

    data_type = np.float32
    print(f"Data type: {data_type}")

    structures = dataframe["structure"]
    targets = dataframe["n"].to_numpy(dtype=data_type).reshape(-1, 1)  # 2D numpy array

    # === SOAP ===
    print("\nGenerating SOAP feature matrix...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)
    memory = soap_featurizer.estimate_dense_feature_matrix_memory_gb(
        len(structures), data_type
    )
    print(f"Estimated SOAP feature matrix memory: {memory:.2f} GB")
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(
        structures, dtype=data_type
    )
    # X_soap = X_soap.astype(data_type)
    print(f"SOAP feature count: {X_soap.shape[1]}")

    # === Composition ===
    print("\nGenerating composition feature matrix and label...")
    compositions = fcomp.extract_compositions_from_structures(structures)
    composition_featurizer = fcomp.create_composition_featurizer()
    X_composition, composition_feature_names = (
        fcomp.create_composition_feature_matrix_and_labels(
            compositions, composition_featurizer
        )
    )
    # X_composition = X_composition.astype(data_type)
    print(f"Composition feature count: {X_composition.shape[1]}")

    # === Combined ===
    print("\nConcatenating SOAP and composition feature matrices...")
    X_combined = conc.concatenate_soap_and_composition_feature_matrices(
        X_soap, X_composition
    )
    X_combined = X_combined.astype(data_type)  # converting only here
    input_dim = X_combined_scaled.shape[1]
    print(f"Combined feature count: {input_dim}")

    # === Scaler ===
    print("\nScaling on training data using StandardScaler...")
    target_scaler = prep.fit_target_standard_scaler(
        targets[split.train]
    )  # scaling using train set only
    y_scaled = target_scaler.transform(targets)
    y_test_og = targets[split.test]  # in original refractive-index units

    combined_scaler = prep.fit_feature_standard_scaler(
        X_combined[split.train]
    )  # scaling using train set only
    X_combined_scaled = combined_scaler.transform(X_combined).astype(data_type)

    # === DataLoaders ===
    print("\nCreating DataLoaders...")
    loaders = dl.create_regression_dataloaders(
        features=X_combined_scaled,
        targets=y_scaled,
        train_indices=split.train,
        validation_indices=split.validation,
        test_indices=split.test,
        batch_size=64,
        random_seed=random_seed,
    )

    print("=== Neural Network Training ===")

    # === Overfit tiny subset ===
    print("\nOverfitting tiny subset...")

    rows = 500  # selecting 500 rows

    X_tiny = X_combined_scaled[:rows]
    y_tiny = y_scaled[:rows]

    tiny_dataset = dl.ArrayRegressionDataset(X_tiny, y_tiny)
    tiny_loader = torch.utils.data.DataLoader(tiny_dataset, batch_size=rows)

    print(f"Training on {rows} samples...")

    tiny_model = RegressionMLP(
        input_dimension=X_tiny.shape[1],
    )

    fit_neural_network(
        model=tiny_model,
        training_dataloader=tiny_loader,
        number_of_epochs=50,
    )

    print("\nFull training...")
    model = RegressionMLP(
        input_dimension=input_dim,
    )

    fit_neural_network(
        model=model,
        training_dataloader=loaders.train,
        validation_dataloader=loaders.validation,
    )


if __name__ == "__main__":
    main()
