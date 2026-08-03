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

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
        raise NotImplementedError(
            "Construct the regression MLP and training configuration"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (batch_size, input_dimension) to (batch_size, 1)."""
        raise NotImplementedError("Implement the MLP forward pass")

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
        raise NotImplementedError(
            "Implement prediction and loss calculation shared by all stages"
        )

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
        raise NotImplementedError("Implement the training-step hook")

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
        raise NotImplementedError("Implement the validation-step hook")

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
        raise NotImplementedError("Implement the test-step hook")

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
        raise NotImplementedError("Configure the Adam optimizer")


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
    raise NotImplementedError("Implement one hook-based training epoch")


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
    raise NotImplementedError(
        "Implement hook-based validation or test evaluation"
    )


def fit_neural_network(
    model: RegressionMLP,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
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
    raise NotImplementedError(
        "Implement the Lightning-inspired MLP fitting loop"
    )


def main() -> None:
    """Create the MLP, overfit a tiny subset, and start full training."""
    raise NotImplementedError(
        "Assemble the hook-based neural-network regression workflow"
    )


if __name__ == "__main__":
    main()
