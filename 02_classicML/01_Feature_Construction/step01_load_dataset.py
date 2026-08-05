"""Step 01: load, inspect, and split the dielectric dataset.

Complete each unfinished function by replacing raise NotImplementedError.

Learning goals
--------------
1. Work with a materials dataset containing variable-size crystal structures.
2. Optionally describe its structures, chemical elements, and scalar target.
3. Create reproducible train/validation/test partitions.

A tiny bit of knowledge: dataframe labels and positional split indices
-----------------------------------------------------------------------
Matminer returns a pandas DataFrame. Its external index labels are not
necessarily the same thing as row positions. In this exercise, DatasetSplit
stores integer row positions from 0 to number_of_samples - 1.

Use:

    dataframe.iloc[split.train]

for positional dataframe selection, or convert a column to NumPy first:

    targets = dataframe["n"].to_numpy(dtype=np.float64)
    training_targets = targets[split.train]

Keeping split indices instead of copied dataframes makes it easy to reuse
exactly the same materials for SOAP, composition features, and later models.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset

from sklearn.model_selection import train_test_split


#### [ 1. Load a materials dataset containing crystal structures ] ####
def load_dielectric_dataset() -> pd.DataFrame:
    """Load and validate the matbench_dielectric dataframe.

    The returned dataframe must contain:

    - structure: one pymatgen.Structure per material;
    - n: the scalar refractive-index target.

    This function is provided so a new student can focus on the exercise rather
    than on the Matminer dataset-loading API.
    """
    dataframe = load_dataset("matbench_dielectric")

    required_columns = {"structure", "n"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "The matbench_dielectric dataset is missing required columns: "
            f"{missing}"
        )

    if dataframe.empty:
        raise ValueError("The matbench_dielectric dataset is empty")

    columns_with_missing_values = [
        column
        for column in sorted(required_columns)
        if dataframe[column].isna().any()
    ]
    if columns_with_missing_values:
        columns = ", ".join(columns_with_missing_values)
        raise ValueError(f"Dataset contains missing values in columns: {columns}")

    return dataframe


def compute_dataset_statistics(
    structures: pd.Series, 
    targets: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
    """Return global statistics and an element-frequency table.

    Requirements
    ------------
    The global statistics should include, at minimum:

    - number of structures;
    - number of unique elements across the dataset;
    - target minimum, maximum, mean, median, and standard deviation;
    - minimum, maximum, mean, and median atoms per structure;
    - minimum, maximum, and mean distinct elements per structure.

    The second output should contain one row per chemical element with:

    - element symbol;
    - number of structures containing that element.

    Beware / remember
    -----------------
    Count each structure once in the element-frequency table, even when several
    atoms of the same element occur in its unit cell.
    """

    n_struct = len(structures) # number of structures
    n_atoms_per_struct = np.array([len(s) for s in structures]) # number of atoms per structure
    unique_elements_per_struct = [set(el.symbol for el in s.composition.elements) for s in structures] # sets of unique elements per struct
    n_distinct_elements = np.array([len(els) for els in unique_elements_per_struct]) # number of distinct elements per structure
    element_freq = pd.Series(unique_elements_per_struct).explode().value_counts().reset_index() # elements frequencies pd.DataFrame

    stats = pd.Series({
        "n_structures": n_struct,
        "n_unique_elements": len(element_freq),
        "target_min": targets.min(),
        "target_max": targets.max(),
        "target_mean": targets.mean(),
        "target_median": targets.median(),
        "target_std": targets.std(),
        "atoms_per_structure_min": n_atoms_per_struct.min(),
        "atoms_per_structure_max": n_atoms_per_struct.max(),
        "atoms_per_structure_mean": n_atoms_per_struct.mean(),
        "atoms_per_structure_median": np.median(n_atoms_per_struct),
        "distinct_elements_per_structure_min": n_distinct_elements.min(),
        "distinct_elements_per_structure_max": n_distinct_elements.max(),
        "distinct_elements_per_structure_mean": n_distinct_elements.mean(),
    })

    element_freq.columns = ["element", "n_structures"] # renaming columns

    return stats, element_freq

#### [ 2. Create reproducible train/validation/test partitions ] ####
@dataclass
class DatasetSplit:
    """Suggested container for three non-overlapping index partitions.

    train, validation, and test contain positional integer row indices rather
    than dataframe copies.

    This dataclass is only one possible design. A dictionary, tuple, or another
    clearly documented container is acceptable if downstream code can access
    all partitions consistently.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def create_dataset_split(
    n_samples: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create reproducible, non-overlapping dataset index partitions.

    Requirements
    ------------
    - Use sklearn.model_selection.train_test_split instead of manually
      shuffling indices.
    - Pass random_seed through the random_state argument.
    - Return train and test indices, with validation optionally empty when
      validation_fraction == 0.
    - One possible strategy is to separate test first and then split the
      remaining indices into train and validation.
    - If using two calls, convert validation_fraction into a fraction of the
      data remaining after the test split.
    - Validate n_samples and both requested fractions.
    - Ensure every position appears exactly once across the partitions.
    - Do not depend on external dataframe index labels.

    Note
    ----
    Returning DatasetSplit is suggested, not required. If you choose a
    different representation, document it and update downstream files
    consistently.
    """
    # input validation
    if not isinstance(n_samples, int):
        raise TypeError(f"n_samples must be a integer (int), input: {type(n_samples)}") #??
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Fractions must be non negative.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("The sum of validation and test must be lower than 1.")    

    indices = np.arange(n_samples) # array of indices

    # test set indices
    if test_fraction > 0:
      train_val_indices, test_indices = train_test_split(indices, test_size=test_fraction, random_state=random_seed)
    else:
      train_val_indices = indices
      test_indices = np.array([], dtype=int)

    # validation set indices
    if validation_fraction > 0:
      validation_relative_fraction = validation_fraction / (1 - test_fraction)
      train_indices, validation_indices = train_test_split(train_val_indices, test_size=validation_relative_fraction, random_state=random_seed)
    else:
      train_indices = train_val_indices
      validation_indices = np.array([], dtype=int)

    return DatasetSplit(train=train_indices, validation=validation_indices, test=test_indices)

def main() -> None:
    """Load, inspect, and split the dataset using the functions above."""
    print("Loading matbench_dielectric dataframe \n")
    dataframe = load_dielectric_dataset()

    print("Computing dataset statistics \n")
    stats, element_freq = compute_dataset_statistics(dataframe["structure"], dataframe["n"])
    print("--- Dataset Statistics ---")
    print(stats)
    print("\n--- Top 5 Most Frequent Elements ---")
    print(element_freq.head(5))
    print("\n" + "=" * 40 + "\n")

    print("Creating dataset splits")
    split = create_dataset_split(n_samples=len(dataframe), validation_fraction=0.15, test_fraction=0.15, random_seed=42)


if __name__ == "__main__":
    main()
