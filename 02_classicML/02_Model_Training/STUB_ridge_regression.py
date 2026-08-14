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

import sys
from pathlib import Path
feature_construction = Path(__file__).resolve().parent.parent / "01_Feature_Construction"
if str(feature_construction) not in sys.path:
    sys.path.append(str(feature_construction))

import step01_load_dataset as load
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp
import step03_concatenate_features as conc
import step04_preprocessing as prep

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
    # input validation
    if not isinstance(training_features, np.ndarray) or not isinstance(training_targets, np.ndarray):
        raise TypeError("Training arrays must be a numpy arrays.")
    if training_features.ndim != 2:
        raise ValueError("Training features must be a 2D array.")
    if not np.isfinite(training_features).all() or not np.isfinite(training_targets).all():
        raise ValueError("Training arrays contain non-finite values.")
    if training_features.shape[0] != training_targets.shape[0]:
        raise ValueError("Training arrays are not alligned.")
    
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("Alpha must be a positive number.")

    if training_targets.ndim == 1:
        targets_1d = training_targets
    elif training_targets.ndim == 2:
        if training_targets.shape[1] > 1:
            raise ValueError(f"2D targets must be a single column.")
        targets_1d = training_targets.ravel()  # flatten the column (n_sample, 1) to (n_sample,)
    else:
        raise ValueError("Training targets must be a 1D or 2D array.")
    
    # create and fit
    ridge = Ridge(alpha=alpha)

    ridge.fit(training_features, targets_1d)

    return ridge

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
    # input validation
    if not isinstance(targets, np.ndarray) or not isinstance(predictions, np.ndarray):
        raise TypeError("Input arrays must be a numpy arrays.")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Input arrays contain non-finite values.")
    if targets.shape != predictions.shape:
        raise ValueError("Input arrays are not alligned.")
    
    if targets.ndim == 1:
        targets_1d = targets
    elif targets.ndim == 2:
        if targets.shape[1] > 1:
            raise ValueError(f"2D targets must be a single column.")
        targets_1d = targets.ravel()
    else:
        raise ValueError("Targets must be a 1D or 2D array.")

    if predictions.ndim == 1:
        predictions_1d = predictions
    elif predictions.ndim == 2:
        if predictions.shape[1] > 1:
            raise ValueError(f"2D predictions must be a single column.")
        predictions_1d = predictions.ravel()
    else:
        raise ValueError("Predictions must be a 1D or 2D array.")

    # metrics calculation
    mae = mean_absolute_error(targets_1d, predictions_1d)
    mse = mean_squared_error(targets_1d, predictions_1d)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_1d, predictions_1d)
    
    metrics = {
        "Mean Absolute Error" : mae,
        "Mean Squared Error" : mse,
        "Root Mean Squared Error" : rmse,
        "R-squared" : r2,
    }

    return metrics

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
    print("=== Preprocessing ===")
    # === Dataset ===
    print("Loading dataset...")
    dataframe = load.load_dielectric_dataset()

    print("Splitting dataset...")
    n_samples = len(dataframe)
    validation_fraction=0.15
    test_fraction=0.15
    random_seed=42 

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
    targets = dataframe["n"].to_numpy(dtype=data_type).reshape(-1,1) # 2D numpy array

    # === SOAP ===
    print("\nGenerating SOAP feature matrix...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)
    memory = soap_featurizer.estimate_dense_feature_matrix_memory_gb(len(structures), data_type)
    print(f"Estimated SOAP feature matrix memory: {memory:.2f} GB")
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(structures, dtype = data_type)
    X_soap = X_soap.astype(data_type)
    print(f"SOAP feature count: {X_soap.shape[1]}")

    # === Composition ===
    print("\nGenerating composition feature matrix and label...")
    compositions = fcomp.extract_compositions_from_structures(structures)
    composition_featurizer = fcomp.create_composition_featurizer()
    X_composition, composition_feature_names = fcomp.create_composition_feature_matrix_and_labels(
        compositions, 
        composition_featurizer
    )
    X_composition = X_composition.astype(data_type)
    print(f"Composition feature count: {X_composition.shape[1]}")

    # === Combined ===
    print("\nConcatenating SOAP and composition feature matrices...")
    X_combined = conc.concatenate_soap_and_composition_feature_matrices(X_soap, X_composition)
    X_combined = X_combined.astype(data_type)
    print(f"Combined feature count: {X_combined.shape[1]}")  

    # === Scaler ===
    print("\nScaling on training data using StandardScaler...")
    target_scaler = prep.fit_target_standard_scaler(targets[split.train])
    y_train = target_scaler.transform(targets[split.train])
    y_val_og = targets[split.validation] # in original refractive-index units
    y_test_og = targets[split.test] # in original refractive-index units

    soap_scaler = prep.fit_feature_standard_scaler(X_soap[split.train])
    X_soap_train = soap_scaler.transform(X_soap[split.train]).astype(data_type)
    X_soap_val = soap_scaler.transform(X_soap[split.validation]).astype(data_type)
    X_soap_test = soap_scaler.transform(X_soap[split.test]).astype(data_type)

    composition_scaler = prep.fit_feature_standard_scaler(X_composition[split.train])
    X_composition_train = composition_scaler.transform(X_composition[split.train]).astype(data_type)
    X_composition_val = composition_scaler.transform(X_composition[split.validation]).astype(data_type)
    X_composition_test = composition_scaler.transform(X_composition[split.test]).astype(data_type)

    combined_scaler = prep.fit_feature_standard_scaler(X_combined[split.train])
    X_combined_train = combined_scaler.transform(X_combined[split.train]).astype(data_type)
    X_combined_val = combined_scaler.transform(X_combined[split.validation]).astype(data_type)
    X_combined_test = combined_scaler.transform(X_combined[split.test]).astype(data_type)

    # === Tuning ===
    print("\n=== Alpha Tuning ===")
    alpha_test = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    print("SOAP features...")
    soap_best_alpha = None
    soap_best_val_mae = float("inf")
    soap_best_model = None
    
    for alpha in alpha_test:
        soap_model = fit_ridge_regression(X_soap_train, y_train, alpha)
        soap_pred = soap_model.predict(X_soap_val)
        soap_pred_og = target_scaler.inverse_transform(soap_pred.reshape(-1, 1)) # in original refractive-index units
        metrics_soap = calculate_regression_metrics(y_val_og, soap_pred_og)
        
        if metrics_soap["Mean Absolute Error"] < soap_best_val_mae:
            soap_best_alpha = alpha
            soap_best_val_mae = metrics_soap["Mean Absolute Error"]
            soap_best_model = soap_model
        
        print(f"    alpha = {alpha} : MAE = {metrics_soap['Mean Absolute Error']:.4f}")
    print(f"Best alpha: {soap_best_alpha} : MAE = {soap_best_val_mae:.4f}")

    print("\nComposition features...")
    comp_best_alpha = None
    comp_best_val_mae = float("inf")
    comp_best_model = None

    for alpha in alpha_test:
        comp_model = fit_ridge_regression(X_composition_train, y_train, alpha)
        comp_pred = comp_model.predict(X_composition_val)
        comp_pred_og = target_scaler.inverse_transform(comp_pred.reshape(-1, 1)) # in original refractive-index units
        metrics_comp = calculate_regression_metrics(y_val_og, comp_pred_og)

        if metrics_comp["Mean Absolute Error"] < comp_best_val_mae:
            comp_best_alpha = alpha
            comp_best_val_mae = metrics_comp["Mean Absolute Error"]
            comp_best_model = comp_model

        print(f"    alpha = {alpha} : MAE = {metrics_comp['Mean Absolute Error']:.4f}")
    print(f"Best alpha: {comp_best_alpha} : MAE = {comp_best_val_mae:.4f}")

    print("\nCombined features...")
    comb_best_alpha = None
    comb_best_val_mae = float("inf")
    comb_best_model = None

    for alpha in alpha_test:
        comb_model = fit_ridge_regression(X_combined_train, y_train, alpha)
        comb_pred = comb_model.predict(X_combined_val)
        comb_pred_og = target_scaler.inverse_transform(comb_pred.reshape(-1, 1)) # in original refractive-index units
        metrics_comb = calculate_regression_metrics(y_val_og, comb_pred_og)

        if metrics_comb["Mean Absolute Error"] < comb_best_val_mae:
            comb_best_alpha = alpha
            comb_best_val_mae = metrics_comb["Mean Absolute Error"]
            comb_best_model = comb_model

        print(f"    alpha = {alpha} : MAE = {metrics_comb['Mean Absolute Error']:.4f}")
    print(f"Best alpha: {comb_best_alpha} : MAE = {comb_best_val_mae:.4f}")

    # === Testing ===
    print("\n=== Testing ===")

    print("SOAP features...")
    soap_test_pred = soap_best_model.predict(X_soap_test)
    soap_test_pred_og = target_scaler.inverse_transform(soap_test_pred.reshape(-1,1)) # in original refractive-index units
    metrics_soap = calculate_regression_metrics(y_test_og, soap_test_pred_og)
    print(metrics_soap)

    print("\nComposition features...")
    comp_test_pred = comp_best_model.predict(X_composition_test)
    comp_test_pred_og = target_scaler.inverse_transform(comp_test_pred.reshape(-1,1)) # in original refractive-index units
    metrics_comp = calculate_regression_metrics(y_test_og, comp_test_pred_og)
    print(metrics_comp)

    print("\nCombined features...")
    comb_test_pred = comb_best_model.predict(X_combined_test)
    comb_test_pred_og = target_scaler.inverse_transform(comb_test_pred.reshape(-1,1)) # in original refractive-index units
    metrics_comb = calculate_regression_metrics(y_test_og, comb_test_pred_og)
    print(metrics_comb)

if __name__ == "__main__":
    main()
