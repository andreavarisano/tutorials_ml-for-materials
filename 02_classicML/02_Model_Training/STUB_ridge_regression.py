"""ROUGH DRAFT: establish a Ridge-regression baseline.

Ridge should be trained before the neural network. It is fast, interpretable,
and often difficult to beat when fixed descriptors and limited data are used.

Please use the same feature representations, split indices, preprocessing rules, and
metrics used for the neural network.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RidgeValidationResult:
    """Best validation result and fitted Ridge model."""

    model: Ridge
    alpha: float
    validation_mae: float


def fit_ridge_regression(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    alpha: float,
    ) -> Ridge:
    """Fit one Ridge model using a specified regularization strength.

    TODO
    ----
    - Validate finite, aligned training arrays.
    - Require a positive alpha.
    - Convert a single-column target to the one-dimensional shape expected for
      scalar regression.
    - Create sklearn.linear_model.Ridge.
    - Fit only on training data.
    - Return the fitted estimator.
    """
    raise NotImplementedError("Validate inputs and fit one Ridge model")

def calculate_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for aligned scalar predictions.

    TODO
    ----
    - Validate shapes and finite values.
    - Flatten single-column arrays consistently.
    - Use mean_absolute_error and mean_squared_error.
    - Calculate RMSE as the square root of MSE.
    - Use r2_score.
    """
    raise NotImplementedError("Calculate standard scalar-regression metrics")


def main() -> None:
    """Compare Ridge across composition, SOAP, and combined representations.

    TODO
    ----
    - Use identical train/validation/test materials for every representation.
    - Tune alpha separately for each representation.
    - Do not use test data during model or alpha selection.
    - Report final metrics in original refractive-index units.
    - Record feature count, selected alpha, and preprocessing configuration.
    """
    raise NotImplementedError("Assemble the Ridge baseline comparison")


if __name__ == "__main__":
    main()
