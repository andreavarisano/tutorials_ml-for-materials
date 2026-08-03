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
    raise NotImplementedError(
        "Validate row alignment and concatenate feature columns"
    )


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
    raise NotImplementedError(
        "Create ordered names for SOAP and composition columns"
    )


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
    raise NotImplementedError(
        "Prepare aligned composition, SOAP, and combined feature matrices"
    )


if __name__ == "__main__":
    main()
