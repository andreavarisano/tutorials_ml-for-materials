"""Step 04: standardize targets and feature matrices without leakage."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import step01_load_dataset as load
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp
import step03_concatenate_features as conc


#### [ 1. Fit a scaler for the scalar regression target ] ####
def _validate_and_reshape_target_array(targets: np.ndarray) -> np.ndarray:
    """Validate and reshape the target array to 2D."""
    if targets.size == 0:
        raise ValueError("The target array cannot be empty.")
    if not np.issubdtype(targets.dtype, np.number):
        raise TypeError("The target array must be numeric.")
    if not np.isfinite(targets).all():
        raise ValueError("The target array contains NaN or Inf values.")

    if targets.ndim == 1:
        return targets.reshape(-1, 1)
    elif targets.ndim == 2:
        if targets.shape[1] > 1:
            raise ValueError("The target array must be a single column.")
        return targets
    else:
        raise ValueError(
            "The target array must be a one-dimensional or two-dimensional array."
        )


def fit_target_standard_scaler(
    training_targets: np.ndarray,
) -> StandardScaler:
    """Fit and return a StandardScaler for training targets only."""
    training_targets = _validate_and_reshape_target_array(training_targets)

    # fitting
    scaler = StandardScaler()
    scaler.fit(training_targets)

    return scaler


def fit_target_robust_scaler(
    training_targets: np.ndarray,
) -> RobustScaler:
    """Fit and return a RobustScaler for training targets only."""
    training_targets = _validate_and_reshape_target_array(training_targets)

    # fitting
    scaler = RobustScaler()
    scaler.fit(training_targets)

    return scaler


#### [ 2. Fit a scaler for a feature representation ] ####
def _validate_features(training_features: np.ndarray) -> None:
    """Validate the training feature matrix."""
    if training_features.size == 0:
        raise ValueError("The training features array cannot be empty.")
    if not np.isfinite(training_features).all():
        raise ValueError("The training features array contains NaN or Inf values.")
    if training_features.ndim != 2:
        raise ValueError("The training features array must be a two-dimensional array")


def fit_feature_standard_scaler(
    training_features: np.ndarray,
) -> StandardScaler:
    """Fit and return a StandardScaler for one training feature matrix.

    Possible extensions
    -------------------
    - Compare scaled and unscaled inputs.
    - Use a scikit-learn Pipeline to bind preprocessing to a model.
    - Fit PCA after scaling, using training rows only.
    """
    _validate_features(training_features)

    # fitting
    scaler = StandardScaler()
    scaler.fit(training_features)

    return scaler


def fit_feature_robust_scaler(
    training_features: np.ndarray,
) -> RobustScaler:
    """Fit and return a RobustScaler for one training feature matrix."""
    _validate_features(training_features)

    # fitting
    scaler = RobustScaler()
    scaler.fit(training_features)

    return scaler


#### ONLY STANDARDSCALER
# def main() -> None:
#     """Fit and apply preprocessing to all prepared representations."""
#     # === Dataset ===
#     print("Loading dataset...")
#     dataframe = load.load_dielectric_dataset()

#     split = load.create_dataset_split(
#         n_samples=len(dataframe),
#         validation_fraction=0.15,
#         test_fraction=0.15,
#         random_seed=42,
#     )

#     structures = dataframe["structure"]
#     targets = dataframe["n"].to_numpy(dtype=np.float64).reshape(-1, 1)  # 2D numpy array

#     # === SOAP ===
#     print("Generating SOAP feature matrix...")
#     sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)
#     soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)
#     X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(structures)

#     # === Composition ===
#     print("Generating composition feature matrix and label...")
#     compositions = fcomp.extract_compositions_from_structures(structures)
#     composition_featurizer = fcomp.create_composition_featurizer()
#     X_composition, composition_feature_names = (
#         fcomp.create_composition_feature_matrix_and_labels(
#             compositions, composition_featurizer
#         )
#     )

#     # === Combined ===
#     print("Concatenating SOAP and composition feature matrices...")
#     X_combined = conc.concatenate_soap_and_composition_feature_matrices(
#         X_soap, X_composition
#     )
#     number_of_soap_features = soap_featurizer.number_of_features_per_atomic_environment
#     combined_names = conc.create_names_for_concatenated_features(
#         number_of_soap_features, composition_feature_names
#     )

#     # === Scaler ===
#     print("Scaling on training data...")
#     target_scaler = fit_target_standard_scaler(targets[split.train])
#     y_train = target_scaler.transform(targets[split.train])
#     y_val = target_scaler.transform(targets[split.validation])
#     y_test = target_scaler.transform(targets[split.test])

#     soap_scaler = fit_feature_standard_scaler(X_soap[split.train])
#     X_soap_train = soap_scaler.transform(X_soap[split.train])
#     X_soap_val = soap_scaler.transform(X_soap[split.validation])
#     X_soap_test = soap_scaler.transform(X_soap[split.test])

#     composition_scaler = fit_feature_standard_scaler(X_composition[split.train])
#     X_composition_train = composition_scaler.transform(X_composition[split.train])
#     X_composition_val = composition_scaler.transform(X_composition[split.validation])
#     X_composition_test = composition_scaler.transform(X_composition[split.test])

#     combined_scaler = fit_feature_standard_scaler(X_combined[split.train])
#     X_combined_train = combined_scaler.transform(X_combined[split.train])
#     X_combined_val = combined_scaler.transform(X_combined[split.validation])
#     X_combined_test = combined_scaler.transform(X_combined[split.test])

#     # === Final checks ===
#     print("\nFinal checks:")
#     np.set_printoptions(suppress=True, precision=4)

#     print(f"Target Train Scaled Mean      : {np.mean(y_train):.4f}")
#     print(f"SOAP Train Scaled Mean        : {np.mean(X_soap_train):.4f}")
#     print(f"Composition Train Scaled Mean : {np.mean(X_composition_train):.4f}")
#     print(f"Combined Train Scaled Mean    : {np.mean(X_combined_train):.4f}")
#     print(f"Target Train Scaled Std       : {np.std(y_train):.4f}")
#     print(f"SOAP Train Max Col Std        : {np.max(np.std(X_soap_train, axis=0)):.4f}")
#     print(
#         f"Composition Train Max Col Std : {np.max(np.std(X_composition_train, axis=0)):.4f}"
#     )
#     print(
#         f"Combined Train Max Col Std    : {np.max(np.std(X_combined_train, axis=0)):.4f}"
#     )

#     print("\nMatrix shapes")
#     print(f"X_soap_train                  : {X_soap_train.shape}")
#     print(f"X_composition_train           : {X_composition_train.shape}")
#     print(f"X_combined_train              : {X_combined_train.shape}")
#     print(f"y_train                       : {y_train.shape}")
#     print(f"X_soap_test                   : {X_soap_test.shape}")
#     print(f"X_composition_test            : {X_composition_test.shape}")
#     print(f"X_combined_test               : {X_combined_test.shape}")
#     print(f"y_test                        : {y_test.shape}")
#     print(f"X_soap_val                    : {X_soap_val.shape}")
#     print(f"X_composition_val             : {X_composition_val.shape}")
#     print(f"X_combined_val                : {X_combined_val.shape}")
#     print(f"y_val                         : {y_val.shape}")


#### COMPARISON OF STANDARD AND ROBUST SCALERS
def main() -> None:
    """Fit and apply preprocessing to all prepared representations."""
    # === Dataset ===
    print("Loading dataset...")
    dataframe = load.load_dielectric_dataset()

    dataframe = dataframe.iloc[:250]  # smoke test for speed

    data_type = np.float32

    split = load.create_dataset_split(
        n_samples=len(dataframe),
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
    )

    structures = dataframe["structure"]
    targets = dataframe["n"].to_numpy(dtype=data_type).reshape(-1, 1)  # 2D numpy array

    # === SOAP ===
    print("Generating SOAP feature matrix...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(
        structures, dtype=data_type
    )

    # === Composition ===
    print("Generating composition feature matrix and label...")
    compositions = fcomp.extract_compositions_from_structures(structures)
    composition_featurizer = fcomp.create_composition_featurizer()
    X_composition, composition_feature_names = (
        fcomp.create_composition_feature_matrix_and_labels(
            compositions, composition_featurizer
        )
    )

    # === Combined ===
    print("Concatenating SOAP and composition feature matrices...")
    X_combined = conc.concatenate_soap_and_composition_feature_matrices(
        X_soap, X_composition
    ).astype(data_type)

    number_of_soap_features = soap_featurizer.number_of_features_per_atomic_environment
    combined_names = conc.create_names_for_concatenated_features(
        number_of_soap_features, composition_feature_names
    )

    # === StandardScaler ===
    print("StandardScaling on training data...")
    target_scaler = fit_target_standard_scaler(targets[split.train])
    y_train_std = target_scaler.transform(targets[split.train])
    y_val_std = target_scaler.transform(targets[split.validation])
    y_test_std = target_scaler.transform(targets[split.test])

    soap_scaler = fit_feature_standard_scaler(X_soap[split.train])
    X_soap_train_std = soap_scaler.transform(X_soap[split.train])
    X_soap_val_std = soap_scaler.transform(X_soap[split.validation])
    X_soap_test_std = soap_scaler.transform(X_soap[split.test])

    composition_scaler = fit_feature_standard_scaler(X_composition[split.train])
    X_composition_train_std = composition_scaler.transform(X_composition[split.train])
    X_composition_val_std = composition_scaler.transform(
        X_composition[split.validation]
    )
    X_composition_test_std = composition_scaler.transform(X_composition[split.test])

    combined_scaler = fit_feature_standard_scaler(X_combined[split.train])
    X_combined_train_std = combined_scaler.transform(X_combined[split.train])
    X_combined_val_std = combined_scaler.transform(X_combined[split.validation])
    X_combined_test_std = combined_scaler.transform(X_combined[split.test])

    # === RobustScaler ===
    print("RobustScaling on training data...")

    target_rscaler = fit_target_robust_scaler(targets[split.train])
    y_train_rob = target_rscaler.transform(targets[split.train])
    y_val_rob = target_rscaler.transform(targets[split.validation])
    y_test_rob = target_rscaler.transform(targets[split.test])

    soap_rscaler = fit_feature_robust_scaler(X_soap[split.train])
    X_soap_train_rob = soap_rscaler.transform(X_soap[split.train])
    X_soap_val_rob = soap_rscaler.transform(X_soap[split.validation])
    X_soap_test_rob = soap_rscaler.transform(X_soap[split.test])

    composition_rscaler = fit_feature_robust_scaler(X_composition[split.train])
    X_composition_train_rob = composition_rscaler.transform(X_composition[split.train])
    X_composition_val_rob = composition_rscaler.transform(
        X_composition[split.validation]
    )
    X_composition_test_rob = composition_rscaler.transform(X_composition[split.test])

    combined_rscaler = fit_feature_robust_scaler(X_combined[split.train])
    X_combined_train_rob = combined_rscaler.transform(X_combined[split.train])
    X_combined_val_rob = combined_rscaler.transform(X_combined[split.validation])
    X_combined_test_rob = combined_rscaler.transform(X_combined[split.test])

    # === FINAL CHECKS & COMPARISON ===
    print("\n=== Final Checks: StandardScaler vs RobustScaler ===")
    np.set_printoptions(suppress=True, precision=4)

    # 1. Means
    print(f"\n--- Statistics: Means ---")
    print(f"{'Metric':<30} | {'StandardScaler':<15} | {'RobustScaler':<15}")
    print("-" * 66)
    print(
        f"{'Target Train Mean':<30} | {np.mean(y_train_std):<15.4f} | {np.mean(y_train_rob):<15.4f}"
    )
    print(
        f"{'Target Val Mean':<30} | {np.mean(y_val_std):<15.4f} | {np.mean(y_val_rob):<15.4f}"
    )
    print(
        f"{'Target Test Mean':<30} | {np.mean(y_test_std):<15.4f} | {np.mean(y_test_rob):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'SOAP Train Mean':<30} | {np.mean(X_soap_train_std):<15.4f} | {np.mean(X_soap_train_rob):<15.4f}"
    )
    print(
        f"{'SOAP Val Mean':<30} | {np.mean(X_soap_val_std):<15.4f} | {np.mean(X_soap_val_rob):<15.4f}"
    )
    print(
        f"{'SOAP Test Mean':<30} | {np.mean(X_soap_test_std):<15.4f} | {np.mean(X_soap_test_rob):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'Comp. Train Mean':<30} | {np.mean(X_composition_train_std):<15.4f} | {np.mean(X_composition_train_rob):<15.4f}"
    )
    print(
        f"{'Comp. Val Mean':<30} | {np.mean(X_composition_val_std):<15.4f} | {np.mean(X_composition_val_rob):<15.4f}"
    )
    print(
        f"{'Comp. Test Mean':<30} | {np.mean(X_composition_test_std):<15.4f} | {np.mean(X_composition_test_rob):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'Combined Train Mean':<30} | {np.mean(X_combined_train_std):<15.4f} | {np.mean(X_combined_train_rob):<15.4f}"
    )
    print(
        f"{'Combined Val Mean':<30} | {np.mean(X_combined_val_std):<15.4f} | {np.mean(X_combined_val_rob):<15.4f}"
    )
    print(
        f"{'Combined Test Mean':<30} | {np.mean(X_combined_test_std):<15.4f} | {np.mean(X_combined_test_rob):<15.4f}"
    )

    # 2. Standard Deviations
    print(f"\n--- Statistics: Standard Deviations ---")
    print(f"{'Metric':<30} | {'StandardScaler':<15} | {'RobustScaler':<15}")
    print("-" * 66)
    print(
        f"{'Target Train Std':<30} | {np.std(y_train_std):<15.4f} | {np.std(y_train_rob):<15.4f}"
    )
    print(
        f"{'Target Val Std':<30} | {np.std(y_val_std):<15.4f} | {np.std(y_val_rob):<15.4f}"
    )
    print(
        f"{'Target Test Std':<30} | {np.std(y_test_std):<15.4f} | {np.std(y_test_rob):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'SOAP Train Max Col Std':<30} | {np.max(np.std(X_soap_train_std, axis=0)):<15.4f} | {np.max(np.std(X_soap_train_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'SOAP Val Max Col Std':<30} | {np.max(np.std(X_soap_val_std, axis=0)):<15.4f} | {np.max(np.std(X_soap_val_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'SOAP Test Max Col Std':<30} | {np.max(np.std(X_soap_test_std, axis=0)):<15.4f} | {np.max(np.std(X_soap_test_rob, axis=0)):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'Comp. Train Max Col Std':<30} | {np.max(np.std(X_composition_train_std, axis=0)):<15.4f} | {np.max(np.std(X_composition_train_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'Comp. Val Max Col Std':<30} | {np.max(np.std(X_composition_val_std, axis=0)):<15.4f} | {np.max(np.std(X_composition_val_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'Comp. Test Max Col Std':<30} | {np.max(np.std(X_composition_test_std, axis=0)):<15.4f} | {np.max(np.std(X_composition_test_rob, axis=0)):<15.4f}"
    )
    print("-" * 66)
    print(
        f"{'Combined Train Max Col Std':<30} | {np.max(np.std(X_combined_train_std, axis=0)):<15.4f} | {np.max(np.std(X_combined_train_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'Combined Val Max Col Std':<30} | {np.max(np.std(X_combined_val_std, axis=0)):<15.4f} | {np.max(np.std(X_combined_val_rob, axis=0)):<15.4f}"
    )
    print(
        f"{'Combined Test Max Col Std':<30} | {np.max(np.std(X_combined_test_std, axis=0)):<15.4f} | {np.max(np.std(X_combined_test_rob, axis=0)):<15.4f}"
    )

    # 3. Shapes
    print(f"\n--- Matrix Shapes ---")
    print(f"{'Array Name':<25} | {'StandardScaler':<18} | {'RobustScaler':<18}")
    print("-" * 66)

    # Train
    print(
        f"{'X_soap_train':<25} | {str(X_soap_train_std.shape):<18} | {str(X_soap_train_rob.shape):<18}"
    )
    print(
        f"{'X_composition_train':<25} | {str(X_composition_train_std.shape):<18} | {str(X_composition_train_rob.shape):<18}"
    )
    print(
        f"{'X_combined_train':<25} | {str(X_combined_train_std.shape):<18} | {str(X_combined_train_rob.shape):<18}"
    )
    print(
        f"{'y_train':<25} | {str(y_train_std.shape):<18} | {str(y_train_rob.shape):<18}"
    )

    # Validation
    print(
        f"{'X_soap_val':<25} | {str(X_soap_val_std.shape):<18} | {str(X_soap_val_rob.shape):<18}"
    )
    print(
        f"{'X_composition_val':<25} | {str(X_composition_val_std.shape):<18} | {str(X_composition_val_rob.shape):<18}"
    )
    print(
        f"{'X_combined_val':<25} | {str(X_combined_val_std.shape):<18} | {str(X_combined_val_rob.shape):<18}"
    )
    print(f"{'y_val':<25} | {str(y_val_std.shape):<18} | {str(y_val_rob.shape):<18}")

    # Test
    print(
        f"{'X_soap_test':<25} | {str(X_soap_test_std.shape):<18} | {str(X_soap_test_rob.shape):<18}"
    )
    print(
        f"{'X_composition_test':<25} | {str(X_composition_test_std.shape):<18} | {str(X_composition_test_rob.shape):<18}"
    )
    print(
        f"{'X_combined_test':<25} | {str(X_combined_test_std.shape):<18} | {str(X_combined_test_rob.shape):<18}"
    )
    print(f"{'y_test':<25} | {str(y_test_std.shape):<18} | {str(y_test_rob.shape):<18}")


if __name__ == "__main__":
    main()
