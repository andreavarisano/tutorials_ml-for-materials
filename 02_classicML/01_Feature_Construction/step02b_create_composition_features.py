"""Step 02b: create composition-only descriptors with Matminer."""

from typing import Sequence

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition, Structure

from step01_load_dataset import load_dielectric_dataset


#### [ 1. Extract compositions without changing structure order ] ####
def _validate_structure_collection(structures: Sequence[Structure] | pd.Series) -> None:
    """Validate the structure collection."""
    if len(structures) == 0:
        raise ValueError("The structure collection is empty.")

    for structure in structures:
        if not isinstance(structure, Structure):
            raise ValueError(f"All structures must be a pymatgen Structure.")


def extract_compositions_from_structures(
    structures: Sequence[Structure] | pd.Series,
) -> list[Composition]:
    """Return one pymatgen Composition object for every structure."""
    _validate_structure_collection(structures)

    return [structure.composition for structure in structures]


#### [ 2. Configure the Matminer composition featurizer ] ####
def create_composition_featurizer() -> MultipleFeaturizer:
    """Create the composition descriptor used in this exercise.

    Possible extensions
    -------------------
    - Add ValenceOrbital for explicit valence-electron statistics.
    - Add BandCenter as a simple electronegativity-derived feature.
    - Compare with ElementFraction and discuss its species-dependent size.
    """

    return MultipleFeaturizer(
        [
            Stoichiometry(),
            ElementProperty.from_preset("magpie"),
        ]
    )


#### [ 3. Calculate the composition feature matrix and labels ] ####
def _validate_feature_matrix(
    matrix: np.ndarray, expected_rows: int, expected_cols: int
) -> None:
    """Validate the composition feature matrix."""
    if matrix.ndim != 2:
        raise ValueError("The matrix does not have two dimensions.")

    if not np.isfinite(matrix).all():
        raise ValueError("The matrix does not have finite values.")

    if matrix.shape[0] != expected_rows:
        raise ValueError(
            "The number of rows does not match the number of compositions."
        )

    if matrix.shape[1] != expected_cols:
        raise ValueError(
            "The number of columns does not match the number of feature names."
        )


def create_composition_feature_matrix_and_labels(
    compositions: Sequence[Composition],
    composition_featurizer: MultipleFeaturizer,
) -> tuple[np.ndarray, list[str]]:
    """Calculate one named composition-feature row per material."""
    # input validation
    if len(compositions) == 0:
        raise ValueError("The composition collection is empty.")

    feature_rows = composition_featurizer.featurize_many(
        compositions,
        ignore_errors=False,
    )
    feature_names = composition_featurizer.feature_labels()

    feature_matrix = np.array(feature_rows)
    _validate_feature_matrix(
        feature_matrix,
        expected_rows=len(compositions),
        expected_cols=len(feature_names),
    )

    return feature_matrix, feature_names


def main() -> None:
    """Run the complete composition-feature construction exercise.

    Requirements
    ------------
    - Load the structures using Step 01.
    - Extract one composition per structure without changing row order.
    - Create the composition featurizer.
    - Calculate feature values and labels.
    - Print and validate matrix shape, feature count, and a few feature names.
    """
    print("Loading dataset...")
    dataframe = load_dielectric_dataset()

    print("Extracting compositions...")
    structures = dataframe["structure"]
    n_structures = len(structures)
    compositions = extract_compositions_from_structures(structures)

    print("Creating the composition featurizer...")
    composition_featurizer = create_composition_featurizer()

    print("Calculating the composition feature matrix and labels...")
    feature_matrix, feature_names = create_composition_feature_matrix_and_labels(
        compositions, composition_featurizer
    )
    # The shape is validated inside the function, but we can print it for user information.
    print(
        f"Final feature matrix shape {feature_matrix.shape} matched the expected shape {(n_structures, len(feature_names))}\n"
    )

    print(f"Total number of features generated: {len(feature_names)}")
    print("First 10 feature names:")
    for name in feature_names[:10]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
