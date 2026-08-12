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

import step01_load_dataset as load
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp
import step03_concatenate_features as conc


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
    # input validation
    if training_targets.size == 0:
        raise ValueError("The training targets array cannot be empty.")
    if not np.issubdtype(training_targets.dtype, np.number):
        raise TypeError("The training targets array must be numeric.")
    if not np.isfinite(training_targets).all():
        raise ValueError("The training targets array contains NaN or Inf values.")

    # reshaping
    if training_targets.ndim == 1:
        training_targets_2d = training_targets.reshape(-1, 1)
    elif training_targets.ndim == 2:
        if training_targets.shape[1] > 1:
            raise ValueError("The training targets array must be a single column.")
        training_targets_2d = training_targets
    else:
        raise ValueError("The training targets array must be a one-dimensional or two-dimensional array.")

    # fitting
    scaler = StandardScaler()
    scaler.fit(training_targets_2d)

    return scaler


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
    # input validation
    if training_features.size == 0:
        raise ValueError("The training features array cannot be empty.")
    if not np.isfinite(training_features).all():
        raise ValueError("The training features array contains NaN or Inf values.")
    if training_features.ndim != 2:
        raise ValueError("The training features array must be a two-dimensional array")

    # fitting
    scaler = StandardScaler()
    scaler.fit(training_features)

    return scaler


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
    # === Dataset ===
    print("Loading dataset...")
    dataframe = load.load_dielectric_dataset()
        
    split = load.create_dataset_split(
        n_samples=len(dataframe),
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
    )
    
    structures = dataframe["structure"]
    targets = dataframe["n"].to_numpy(dtype=np.float64).reshape(-1,1) # 2D numpy array

    # === SOAP ===
    print("Generating SOAP feature matrix...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(structures)

    # === Composition ===
    print("Generating composition feature matrix and label...")
    compositions = fcomp.extract_compositions_from_structures(structures)
    composition_featurizer = fcomp.create_composition_featurizer()
    X_composition, composition_feature_names = fcomp.create_composition_feature_matrix_and_labels(
        compositions, 
        composition_featurizer
    )

    # === Combined ===
    print("Concatenating SOAP and composition feature matrices...")
    X_combined = conc.concatenate_soap_and_composition_feature_matrices(X_soap, X_composition)
    number_of_soap_features = soap_featurizer.number_of_features_per_atomic_environment
    combined_names = conc.create_names_for_concatenated_features(number_of_soap_features, composition_feature_names)

    # === Scaler ===
    print("Scaling on training data...")
    target_scaler = fit_target_standard_scaler(targets[split.train])
    y_train = target_scaler.transform(targets[split.train])
    y_val = target_scaler.transform(targets[split.validation])
    y_test = target_scaler.transform(targets[split.test])

    soap_scaler = fit_feature_standard_scaler(X_soap[split.train])
    X_soap_train = soap_scaler.transform(X_soap[split.train])
    X_soap_val = soap_scaler.transform(X_soap[split.validation])
    X_soap_test = soap_scaler.transform(X_soap[split.test])

    composition_scaler = fit_feature_standard_scaler(X_composition[split.train])
    X_composition_train = composition_scaler.transform(X_composition[split.train])
    X_composition_val = composition_scaler.transform(X_composition[split.validation])
    X_composition_test = composition_scaler.transform(X_composition[split.test])

    combined_scaler = fit_feature_standard_scaler(X_combined[split.train])
    X_combined_train = combined_scaler.transform(X_combined[split.train])
    X_combined_val = combined_scaler.transform(X_combined[split.validation])
    X_combined_test = combined_scaler.transform(X_combined[split.test])

    # === Final checks ===
    print("\nFinal checks:")
    np.set_printoptions(suppress=True, precision=4)

    print(f"Target Train Scaled Mean      : {np.mean(y_train):.4f}")
    print(f"SOAP Train Scaled Mean        : {np.mean(X_soap_train):.4f}")
    print(f"Composition Train Scaled Mean : {np.mean(X_composition_train):.4f}")
    print(f"Combined Train Scaled Mean    : {np.mean(X_combined_train):.4f}")
    print(f"Target Train Scaled Std       : {np.std(y_train):.4f}")
    print(f"SOAP Train Max Col Std        : {np.max(np.std(X_soap_train, axis=0)):.4f}")
    print(f"Composition Train Max Col Std : {np.max(np.std(X_composition_train, axis=0)):.4f}")
    print(f"Combined Train Max Col Std    : {np.max(np.std(X_combined_train, axis=0)):.4f}")
    
    print("\nMatrix shapes")
    print(f"X_soap_train                  : {X_soap_train.shape}")
    print(f"X_composition_train           : {X_composition_train.shape}")
    print(f"X_combined_train              : {X_combined_train.shape}")
    print(f"y_train                       : {y_train.shape}")
    print(f"X_soap_test                   : {X_soap_test.shape}")
    print(f"X_composition_test            : {X_composition_test.shape}")
    print(f"X_combined_test               : {X_combined_test.shape}")
    print(f"y_test                        : {y_test.shape}")
    print(f"X_soap_val                    : {X_soap_val.shape}")
    print(f"X_composition_val             : {X_composition_val.shape}")
    print(f"X_combined_val                : {X_combined_val.shape}")
    print(f"y_val                         : {y_val.shape}")


if __name__ == "__main__":
    main()
