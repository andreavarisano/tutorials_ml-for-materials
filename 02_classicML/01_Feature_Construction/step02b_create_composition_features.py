"""Step 02b: create composition-only descriptors with Matminer.

Complete each function by replacing raise NotImplementedError.

Learning goals
--------------
1. Convert crystal structures into pymatgen Composition objects.
2. Configure interpretable Matminer composition featurizers.
3. Create a finite, named composition-feature matrix.
4. Preserve structure-row order for later comparison with SOAP.


A tiny bit of knowledge: composition and structure describe different things
----------------------------------------------------------------------------
Composition descriptors use only which elements are present and their relative
amounts. They are global, structure-independent descriptors: they do not use
atomic coordinates, bond lengths, bond angles, or local atomic environments.

Consequently, two polymorphs with the same composition receive identical
composition descriptors.

SOAP instead begins from the local environment around each atom and contains
both chemical and structural information. Pooling converts those local
descriptors into a global vector, but it does not erase all local-environment
information: it aggregates it. It does remove site correspondence and may lose
information about the distribution of environments.

SOAP and composition descriptors are therefore not completely independent,
because SOAP already contains chemical information. A later exercise will test
whether explicit global composition statistics add useful information beyond
pooled SOAP.


A practical Matminer composition workflow
------------------------------------------
Constructing the composition representation has two main stages.

1. Configure the Matminer featurizer:

       composition_featurizer = MultipleFeaturizer([
           Stoichiometry(),
           ElementProperty.from_preset("magpie"),
       ])

   Stoichiometry describes concentration patterns. The Magpie preset calculates
   statistics of tabulated elemental properties, such as composition-weighted
   means and ranges.

2. Calculate the feature rows and their names:

       compositions = extract_compositions_from_structures(structures)
       feature_rows = composition_featurizer.featurize_many(compositions)
       feature_names = composition_featurizer.feature_labels()

Always query feature_labels() rather than hard-coding the descriptor dimension.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition, Structure

from step01_load_dataset import load_dielectric_dataset


#### [ 1. Extract compositions without changing structure order ] ####
def extract_compositions_from_structures(
    structures: Sequence[Structure] | pd.Series,
) -> list[Composition]:
    """Return one pymatgen Composition object for every structure.

    Requirements
    ------------
    - Reject an empty collection.
    - Check that every entry is a pymatgen Structure.
    - Extract structure.composition.
    - Preserve the original row order.
    - Return exactly one Composition per input structure.

    Beware / remember
    -----------------
    Do not sort structures or deduplicate compositions. Different structures
    may intentionally share a composition, for example polymorphs.

    Advice
    ------
    Start with a few structures and print their reduced formulas beside the
    corresponding dataframe row positions.
    """
    # input validation
    if len(structures) == 0:
      raise ValueError("The structure collection is empty.")

    # list of compositions
    compositions = []

    for i, structure in enumerate(structures):

      # input validation
      if not (isinstance(structure, Structure)):
        raise ValueError("The structure must be a pymatgen Structure.")

      compositions.append(structure.composition)
    return compositions


#### [ 2. Configure the Matminer composition featurizer ] ####
def create_composition_featurizer() -> MultipleFeaturizer:
    """Create the composition descriptor used in this exercise.

    Requirements
    ------------
    - Construct Stoichiometry with its documented default settings.
    - Construct ElementProperty from the "magpie" preset.
    - Combine them in a MultipleFeaturizer, in that order.
    - Return the configured MultipleFeaturizer.

    Beware / remember
    -----------------
    Feature order is part of the representation. Changing featurizer order or
    settings creates a different model input and must be recorded.

    Advice
    ------
    Print feature_labels() immediately after construction. Also inspect and
    retain the scientific references returned by citations().

    Possible extensions
    -------------------
    - Add ValenceOrbital for explicit valence-electron statistics.
    - Add BandCenter as a simple electronegativity-derived feature.
    - Compare with ElementFraction and discuss its species-dependent size.
    """
    stoichiometry = Stoichiometry()
    element_property = ElementProperty.from_preset("magpie")
    composition_featurizer = MultipleFeaturizer([
        stoichiometry,
        element_property,
    ])
    return composition_featurizer


#### [ 3. Calculate the composition feature matrix and labels ] ####
def create_composition_feature_matrix_and_labels(
    compositions: Sequence[Composition],
    composition_featurizer: MultipleFeaturizer,
) -> tuple[np.ndarray, list[str]]:
    """Calculate one named composition-feature row per material.

    Requirements
    ------------
    - Reject an empty composition collection.
    - Call composition_featurizer.featurize_many with errors not ignored.
    - Obtain feature names using composition_featurizer.feature_labels().
    - Convert feature rows into a floating-point NumPy array.
    - Require a finite, two-dimensional matrix.
    - Verify the number of rows and the number of named columns.
    - Return the matrix and feature-name list.

    Beware / remember
    -----------------
    Do not silently discard failed structures. Doing so would break alignment
    with SOAP, targets, and split indices.

    Advice
    ------
    Calculate a small slice first. Print feature names beside the first row so
    the generated values remain interpretable.
    """
    stoichiometry = Stoichiometry()
    element_property = ElementProperty.from_preset("magpie")
    composition_featurizer = MultipleFeaturizer([
        stoichiometry,
        element_property,
    ])
    return composition_featurizer


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
    feature_matrix, feature_names = create_composition_feature_matrix_and_labels(compositions, composition_featurizer)
    n_features = len(feature_names)

    exp_shape = (n_structures, n_features)

    if feature_matrix.shape == exp_shape:
        print(f"Final feature matrix shape {feature_matrix.shape} matched the expected shape {exp_shape}\n")
    else:
        raise ValueError(f"Final feature matrix shape {feature_matrix.shape} did not match the expected shape {exp_shape}\n")

    print(f"Total number of features generated: {len(feature_names)}")
    print("First 10 feature names:")
    for name in feature_names[:10]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
