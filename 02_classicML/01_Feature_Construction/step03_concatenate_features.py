"""Step 03: concatenate independently calculated feature blocks.

Complete each function by replacing raise NotImplementedError.

Goal of this exercise
---------------------
Learn to concatenate SOAP and composition feature blocks along the feature axis
while preserving row order.

Concatenation is a simple operation, but it is easy to introduce errors that
invalidate the experiment. We should organize a drinking game around the number
of times we have to say "row order" in this exercise.


Learning goals
--------------
1. Validate two independently calculated feature matrices.
2. Preserve material-row identity and feature names.
3. Concatenate feature blocks along the feature axis.
4. Prepare composition-only, SOAP-only, and combined representations.

We will compare three representations for the same dataset:

    X_composition
    X_soap
    X_combined = [X_soap | X_composition]

Concatenation may add complementary information, but it may also add irrelevant
or redundant dimensions, increase computational cost, and make overfitting
easier. More features do not automatically produce a better model.


A tiny bit of knowledge: concatenating feature blocks
-----------------------------------------------------
Both matrices must contain the same materials in exactly the same row order:

    X_soap.shape        == (number_of_structures, number_of_soap_features)
    X_composition.shape == (number_of_structures, number_of_composition_features)

Concatenate columns, not rows:

    X_combined = np.concatenate(
        [X_soap, X_composition],
        axis=1,
    )

The result must have shape:

    (
        number_of_structures,
        number_of_soap_features + number_of_composition_features,
    )

Equal row counts are necessary but do not prove equal row identity. Both blocks
must be constructed from the same ordered structure collection. A silent
row-order mismatch would combine information from different materials.
"""

from typing import Sequence

import numpy as np

from step01_load_dataset import load_dielectric_dataset
import step02a_create_soap as fsoap
import step02b_create_composition_features as fcomp 


#### [ 1. Concatenate SOAP and composition columns ] ####
def concatenate_soap_and_composition_feature_matrices(
    soap_features: np.ndarray,
    composition_features: np.ndarray,
) -> np.ndarray:
    """Concatenate aligned SOAP and composition matrices along columns.

    Requirements
    ------------
    - Require two finite, non-empty, two-dimensional arrays.
    - Check that they contain the same number of rows.
    - Concatenate with numpy.concatenate(..., axis=1).
    - Verify the resulting row and column counts.
    - Return the combined matrix without modifying either input.

    Beware / remember
    -----------------
    Matching row counts do not prove matching material order. The caller must
    construct both blocks from the same ordered structure list. Using axis=0
    would append materials instead of features and is incorrect.

    Advice
    ------
    Test the function first with small hand-written matrices whose output is
    obvious.
    """
    # input validation
    if not np.isfinite(soap_features).all() or not np.isfinite(composition_features).all():
        raise ValueError("Input arrays must be finite.")
    if soap_features.size == 0 or composition_features.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if soap_features.ndim != 2 or composition_features.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    n_rows_soap, n_cols_soap = soap_features.shape
    n_rows_comp, n_cols_comp = composition_features.shape

    if n_rows_soap != n_rows_comp:
        raise ValueError("Input arrays must have the same number of rows.")
    
    # concatenation
    combined_features = np.concatenate(
        [soap_features, composition_features],
        axis=1,
    )

    # output validation
    n_rows_exp = n_rows_soap
    n_cols_exp = n_cols_soap + n_cols_comp
    if combined_features.shape != (n_rows_exp, n_cols_exp):
        raise ValueError("Combined array has incorrect shape.")
    
    return combined_features


#### [ 2. Create names for the combined feature columns ] ####
def create_names_for_concatenated_features(
    number_of_soap_features: int,
    composition_feature_names: Sequence[str],
) -> list[str]:
    """Return unique, traceable names for every combined feature column.

    Requirements
    ------------
    - Validate a positive SOAP feature count.
    - Generate deterministic SOAP names such as soap_000000.
    - Prefix Matminer names with composition__.
    - Preserve the matrix-concatenation order.
    - Reject duplicate final names.

    Beware / remember
    -----------------
    SOAP columns may not have simple human-readable physical labels, but stable
    names are still useful for logging, feature selection, and debugging.
    """
    # input validation
    if number_of_soap_features <= 0:
        raise ValueError("Number of SOAP features must be positive.")

    # SOAP names
    soap_names = [f"soap_{i:06d}" for i in range(number_of_soap_features)]

    # composition names
    composition_names = [f"composition__{name}" for name in composition_feature_names]

    # reject duplicate final names
    combined_names = soap_names + composition_names
    if len(combined_names) != len(set(combined_names)):
        raise ValueError("Duplicate feature names found.")

    return combined_names


def main() -> None:
    """Assemble the three raw representations for later preprocessing.

    Requirements
    ------------
    - Obtain SOAP features from Step 02a.
    - Obtain composition features and names from Step 02b.
    - Check that both blocks correspond to the same ordered structures.
    - Create X_combined by concatenating columns.
    - Create and validate the combined feature-name list.
    - Print all three matrix shapes.

    Beware / remember
    -----------------
    This step does not fit scalers. Scaling and leakage prevention belong to
    Step 04.
    """
    print("Loading dataset...")
    dataframe = load_dielectric_dataset()
    structures = dataframe["structure"]

    # === SOAP ===
    print("Extracting species vocabulary...")
    sorted_species = fsoap.collect_sorted_species_list_from_structures(structures)

    print("Creating SOAP featurizer...")
    soap_featurizer = fsoap.SOAPCrystalStructureFeaturizer(species = sorted_species)

    print("Calculating SOAP feature matrix...")
    X_soap = soap_featurizer.create_pooled_feature_matrix_for_structures(structures)

    # === Composition ===
    print("Extracting compositions...")
    compositions = fcomp.extract_compositions_from_structures(structures)

    print("Creating composition featurizer...")
    composition_featurizer = fcomp.create_composition_featurizer()

    print("Calculating composition feature matrix and label...")
    X_composition, composition_feature_names = fcomp.create_composition_feature_matrix_and_labels(compositions, composition_featurizer)

    # === Combined ===
    print("Concatenating SOAP and composition feature matrices...")
    X_combined = concatenate_soap_and_composition_feature_matrices(X_soap, X_composition)

    print("Creating combined feature names...")
    number_of_soap_features = soap_featurizer.number_of_features_per_atomic_environment
    combined_names = create_names_for_concatenated_features(number_of_soap_features, composition_feature_names)

    if X_combined.shape[1] == len(combined_names):
        print("Matrices perfectly aligned and correctly concatenated.")
    else:
        raise ValueError("Matrix columns do not match the number of feature names.")

    # === Final shapes ===
    print(f"\nShape of SOAP feature matrix: {X_soap.shape}")
    print(f"Shape of composition feature matrix: {X_composition.shape}")
    print(f"Shape of combined feature matrix: {X_combined.shape}")

if __name__ == "__main__":
    main()
