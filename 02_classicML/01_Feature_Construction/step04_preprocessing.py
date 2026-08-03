"""Step 04: standardize targets and feature matrices without leakage.

Complete each function by replacing raise NotImplementedError.

Learning goals
--------------
1. Distinguish target scaling from input-feature scaling.
2. Understand the two-dimensional input expected by scikit-learn transformers.
3. Fit every preprocessing object using training samples only.
4. Reuse fitted scalers for validation and test data.
5. Prepare preprocessing that can later be repeated inside cross-validation.


A tiny bit of knowledge: targets and feature matrices have different shapes
----------------------------------------------------------------------------
The refractive-index target usually arrives as a one-dimensional array:

    targets.shape == (number_of_structures,)

StandardScaler expects a two-dimensional matrix, so training targets must become:

    training_targets_2d.shape == (number_of_training_structures, 1)

A feature representation is already two-dimensional:

    training_features.shape == (
        number_of_training_structures,
        number_of_features,
    )

For target scaling, one mean and standard deviation are fitted. For feature
scaling, one mean and standard deviation are fitted independently for every
feature column.


A tiny bit of knowledge: prevent leakage by construction
--------------------------------------------------------
The scaler functions below receive training values directly, rather than the
complete dataset plus split indices:

    target_scaler = fit_target_standard_scaler(targets[split.train])
    soap_scaler = fit_feature_standard_scaler(X_soap[split.train])
    composition_scaler = fit_feature_standard_scaler(
        X_composition[split.train]
    )
    combined_scaler = fit_feature_standard_scaler(X_combined[split.train])

This interface makes it harder for a scaler function to inspect validation or
test rows accidentally. Reuse each fitted object only for its corresponding
representation:

    X_validation_scaled = soap_scaler.transform(X_soap[split.validation])
    X_test_scaled = soap_scaler.transform(X_soap[split.test])

During cross-validation, create and fit fresh scalers inside every fold.


A tiny bit of knowledge: scaling before or after concatenation
--------------------------------------------------------------
StandardScaler treats each column independently. With the same settings,
scaling SOAP and composition separately and then concatenating is numerically
equivalent to concatenating first and scaling every column of the combined
matrix.

This equivalence does not necessarily hold for block normalization, block
weighting, PCA, or other transformations that mix columns. Record the complete
preprocessing order as part of the model configuration.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


#### [ 1. Fit a scaler for the scalar regression target ] ####
def fit_target_standard_scaler(
    training_targets: np.ndarray,
) -> StandardScaler:
    """Fit and return a StandardScaler for training targets only.

    Requirements
    ------------
    - Require a non-empty, numeric, finite NumPy array.
    - Accept a one-dimensional target or a two-dimensional single-column target.
    - Reject arrays with more than one target column.
    - Reshape a one-dimensional target to (number_of_training_samples, 1).
    - Create StandardScaler and fit it only on the supplied training targets.
    - Return the fitted scaler.

    Beware / remember
    -----------------
    Do not pass validation or test targets to this function. Keep the
    two-dimensional shape when calling transform and inverse_transform.

    Advice
    ------
    Inspect scaler.mean_ and scaler.scale_, then verify that transformed
    training targets have approximately zero mean and unit variance.
    """
    raise NotImplementedError(
        "Validate and fit the target scaler on training targets only"
    )


#### [ 2. Fit a scaler for a feature representation ] ####
def fit_feature_standard_scaler(
    training_features: np.ndarray,
) -> StandardScaler:
    """Fit and return a StandardScaler for one training feature matrix.

    Requirements
    ------------
    - Require a non-empty, finite, two-dimensional NumPy array.
    - Create StandardScaler and fit it only on the supplied training rows.
    - Return the fitted scaler.
    - Use a separate fitted scaler for each representation.

    Beware / remember
    -----------------
    Do not pass validation or test rows to this function. StandardScaler
    centering also requires special handling for sparse feature matrices.

    Advice
    ------
    Inspect scaler.mean_ and scaler.scale_. For columns with nonzero variance,
    transformed training values should be approximately centered and have unit
    variance.

    Possible extensions
    -------------------
    - Compare scaled and unscaled inputs.
    - Use a scikit-learn Pipeline to bind preprocessing to a model.
    - Fit PCA after scaling, using training rows only.
    - Compare StandardScaler with RobustScaler.
    """
    raise NotImplementedError(
        "Validate and fit a feature scaler on training rows only"
    )


def main() -> None:
    """Fit and apply preprocessing to all prepared representations.

    Requirements
    ------------
    - Load targets, split indices, and the three matrices from previous steps.
    - Fit the target scaler on targets[split.train].
    - Fit independent SOAP, composition, and combined feature scalers on their
      respective training rows.
    - Transform train, validation, and test with the matching fitted object.
    - Print shapes and simple training-set checks.
    - Keep every scaler together with the representation and model that use it.

    Beware / remember
    -----------------
    Never reuse the SOAP scaler for composition or combined features. Never
    refit a scaler when transforming validation or test data.
    """
    raise NotImplementedError(
        "Fit and apply training-only preprocessing for every representation"
    )


if __name__ == "__main__":
    main()
