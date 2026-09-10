"""Step 03: concatenate independently calculated feature blocks."""

from typing import Sequence

import numpy as np

from step01_load_dataset import load_dielectric_dataset
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp


#### [ 1. Concatenate SOAP and composition columns ] ####
def _validate_matrix_shapes_for_concatenation(
    soap_features: np.ndarray,
    composition_features: np.ndarray,
) -> None:
    """Validate arrays for concatenation."""
    if (
        not np.isfinite(soap_features).all()
        or not np.isfinite(composition_features).all()
    ):
        raise ValueError("Input arrays must be finite.")
    if soap_features.size == 0 or composition_features.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if soap_features.ndim != 2 or composition_features.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")
    if soap_features.ndim != 2 or composition_features.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    # if soap_features.shape[0] != composition_features.shape[0]
    # than np.concatenate(..., axis=1) will raise a ValueError,
    # so we don't need to check that here.


def _validate_concatenated_matrix(
    combined_features: np.ndarray,
    expected_rows: int,
    expected_cols: int,
) -> None:
    """Validate the concatenated feature matrix."""
    if combined_features.shape != (expected_rows, expected_cols):
        raise ValueError("Combined array has incorrect shape.")


def concatenate_soap_and_composition_feature_matrices(
    soap_features: np.ndarray,
    composition_features: np.ndarray,
) -> np.ndarray:
    """Concatenate aligned SOAP and composition matrices along columns."""

    # input validation
    _validate_matrix_shapes_for_concatenation(soap_features, composition_features)

    # concatenation
    combined_features = np.concatenate(
        [soap_features, composition_features],
        axis=1,
    )

    # output validation
    n_rows_exp = soap_features.shape[0]
    n_cols_exp = soap_features.shape[1] + composition_features.shape[1]
    _validate_concatenated_matrix(combined_features, n_rows_exp, n_cols_exp)

    return combined_features


#### [ 2. Create names for the combined feature columns ] ####
def _validate_feature_names(combined_names: Sequence[str]) -> None:
    """Validate the combined feature names."""
    if len(combined_names) != len(set(combined_names)):
        raise ValueError("Duplicate feature names found.")


def create_names_for_concatenated_features(
    number_of_soap_features: int,
    composition_feature_names: Sequence[str],
) -> list[str]:
    """Return unique, traceable names for every combined feature column."""

    if number_of_soap_features <= 0:
        raise ValueError("Number of SOAP features must be positive.")
    # otherwise, the list comprehension below will create an empty list

    soap_names = [f"soap_{i:06d}" for i in range(number_of_soap_features)]
    composition_names = [f"composition__{name}" for name in composition_feature_names]

    combined_names = soap_names + composition_names
    _validate_feature_names(combined_names)

    return combined_names


def main() -> None:
    """Assemble the three raw representations for later preprocessing."""
    print("Loading dataset...")
    dataframe = load_dielectric_dataset()
    structures = dataframe["structure"]

    # === SOAP ===
    print("Extracting species vocabulary...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)

    print("Creating SOAP featurizer...")
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species=sorted_species)

    print("Calculating SOAP feature matrix...")
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(structures)

    # === Composition ===
    print("Extracting compositions...")
    compositions = fcomp.extract_compositions_from_structures(structures)

    print("Creating composition featurizer...")
    composition_featurizer = fcomp.create_composition_featurizer()

    print("Calculating composition feature matrix and label...")
    X_composition, composition_feature_names = (
        fcomp.create_composition_feature_matrix_and_labels(
            compositions, composition_featurizer
        )
    )

    # === Combined ===
    print("Concatenating SOAP and composition feature matrices...")
    X_combined = concatenate_soap_and_composition_feature_matrices(
        X_soap, X_composition
    )

    print("Creating combined feature names...")
    number_of_soap_features = soap_featurizer.number_of_features_per_atomic_environment
    combined_names = create_names_for_concatenated_features(
        number_of_soap_features, composition_feature_names
    )

    # if matrices are not perfectly aligned or uncorrectly concatenated, the concatenation function will raise a ValueError.

    # === Final shapes ===
    print(f"\nShape of SOAP feature matrix: {X_soap.shape}")
    print(f"Shape of composition feature matrix: {X_composition.shape}")
    print(f"Shape of combined feature matrix: {X_combined.shape}")


if __name__ == "__main__":
    main()
