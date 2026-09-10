"""Step 01: load, inspect, and split the dielectric dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset

from sklearn.model_selection import train_test_split


#### [ 1. Load a materials dataset containing crystal structures ] ####
def _validate_dielectric_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate the matbench_dielectric dataframe.

    The dataframe must contain:

    - structure: one pymatgen.Structure per material;
    - n: the scalar refractive-index target.
    """
    required_columns = {"structure", "n"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "The matbench_dielectric dataset is missing required columns: " f"{missing}"
        )

    if dataframe.empty:
        raise ValueError("The matbench_dielectric dataset is empty")

    columns_with_missing_values = [
        column for column in sorted(required_columns) if dataframe[column].isna().any()
    ]
    if columns_with_missing_values:
        columns = ", ".join(columns_with_missing_values)
        raise ValueError(f"Dataset contains missing values in columns: {columns}")


def load_dielectric_dataset() -> pd.DataFrame:
    """Load and validate the matbench_dielectric dataframe."""
    dataframe = load_dataset("matbench_dielectric")

    _validate_dielectric_dataframe(dataframe)

    return dataframe


def compute_dataset_statistics(
    structures: pd.Series,
    targets: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return global statistics and an element-frequency table."""

    n_struct = len(structures)  # number of structures

    n_atoms_per_struct = np.array(
        [len(s) for s in structures]
    )  # number of atoms per structure
    unique_elements_per_struct = [
        set(el.symbol for el in s.composition.elements) for s in structures
    ]  # sets of unique elements per struct
    n_distinct_elements = np.array(
        [len(els) for els in unique_elements_per_struct]
    )  # number of distinct elements per structure

    element_freq = (
        pd.Series(unique_elements_per_struct).explode().value_counts().reset_index()
    )  # elements frequencies pd.DataFrame
    element_freq.columns = ["element", "n_structures"]  # renaming columns

    stats = pd.Series(
        {
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
        }
    )

    return stats, element_freq


#### [ 2. Create reproducible train/validation/test partitions ] ####
@dataclass
class DatasetSplit:
    """Suggested container for three non-overlapping index partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def create_dataset_split(
    n_samples: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create reproducible, non-overlapping dataset index partitions."""
    # input validation
    # arange will raise an error if n_samples is not a positive integer
    # train_test_split will raise an error if fractions are negative

    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("The sum of validation and test must be lower than 1.")

    indices = np.arange(n_samples)  # array of indices

    # test set indices
    if test_fraction > 0:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_fraction, random_state=random_seed
        )
    else:
        train_val_indices = indices
        test_indices = np.array([], dtype=int)

    # validation set indices
    if validation_fraction > 0:
        validation_relative_fraction = validation_fraction / (1 - test_fraction)
        train_indices, validation_indices = train_test_split(
            train_val_indices,
            test_size=validation_relative_fraction,
            random_state=random_seed,
        )
    else:
        train_indices = train_val_indices
        validation_indices = np.array([], dtype=int)

    return DatasetSplit(
        train=train_indices, validation=validation_indices, test=test_indices
    )


def main() -> None:
    """Load, inspect, and split the dataset using the functions above."""
    print("Loading matbench_dielectric dataframe \n")
    dataframe = load_dielectric_dataset()

    print("Computing dataset statistics \n")
    stats, element_freq = compute_dataset_statistics(
        dataframe["structure"], dataframe["n"]
    )
    print("--- Dataset Statistics ---")
    print(stats)
    print("\n--- Top 5 Most Frequent Elements ---")
    print(element_freq.head(5))
    print("\n" + "=" * 40 + "\n")

    print("Creating dataset splits")
    split = create_dataset_split(
        n_samples=len(dataframe),
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
    )


if __name__ == "__main__":
    main()
