"""Step 02a: create fixed-size crystal descriptors using SOAP."""

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from step01_load_dataset import load_dielectric_dataset

PoolingMode = Literal["mean", "sum"]


#### [ 1. Collect the chemical species present in the dataset ] ####
def collect_sorted_species_list_from_structures(
    structures: Sequence[Structure] | pd.Series,
) -> list[str]:
    """Return every chemical species in the structures exactly once."""
    # explicit validation
    if len(structures) == 0:
        raise ValueError("The structure collection is empty.")

    return sorted(
        {
            str(element)
            for structure in structures
            for element in structure.composition.elements
        }
    )


#### [ 2. Configure the SOAP crystal-structure featurizer ] ####
@dataclass
class SOAPCrystalStructureFeaturizer:
    """Configuration and operations needed to create crystal SOAP features.

    Possible extensions
    -------------------
    Add a method that returns the complete configuration as a dictionary for
    logging or serialization.
    """

    species: list[str]
    r_cut: float = 5.0
    n_max: int = 6
    l_max: int = 4
    periodic: bool = True
    sparse: bool = False
    _soap: SOAP = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the settings and create the internal DScribe SOAP object.

        Possible extensions
        -------------------
        - Validate symbols using a periodic-table utility.
        - Add DScribe compression or species-weighting settings.
        - Add a method that compares two featurizer configurations.
        """
        # validation
        if len(self.species) == 0:
            raise ValueError("The species collection is empty.")
        if len(self.species) != len(set(self.species)):
            raise ValueError("The species collection contains duplicates.")
        if self.species != sorted(self.species):
            raise ValueError("The species collection is not sorted.")

        if not (
            isinstance(self.r_cut, (int, float))
            and np.isfinite(self.r_cut)
            and self.r_cut > 0
        ):
            raise ValueError("r_cut must be a finite positive number.")
        if not (isinstance(self.n_max, int) and self.n_max > 0):
            raise ValueError("n_max must be a positive integer.")
        if not (isinstance(self.l_max, int) and self.l_max >= 0):
            raise ValueError("l_max must be a non-negative integer.")
        if not (isinstance(self.periodic, bool) and isinstance(self.sparse, bool)):
            raise ValueError("periodic and sparse must be booleans.")

        # descriptor
        self._soap = SOAP(
            species=self.species,
            periodic=self.periodic,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sparse=self.sparse,
            average="off",
        )

    @classmethod
    def create_featurizer_from_species_list(
        cls,
        species: Sequence[str],
        r_cut: float = 5.0,
        n_max: int = 6,
        l_max: int = 4,
        periodic: bool = True,
        sparse: bool = False,
    ) -> "SOAPCrystalStructureFeaturizer":
        """Create a featurizer from an already collected species list.

        Possible extensions
        -------------------
        Add a create_featurizer_from_configuration_dictionary classmethod for
        recreating an exactly logged descriptor.
        """

        return cls(
            species=list(species),
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            periodic=periodic,
            sparse=sparse,
        )

    @property
    def number_of_features_per_atomic_environment(self) -> int:
        """Return the number of values in one local SOAP descriptor.

        Possible extensions
        -------------------
        Build a small table showing the feature count as species, n_max, and
        l_max change.
        """
        return int(self._soap.get_number_of_features())

    def estimate_dense_feature_matrix_memory_gb(
        self,
        number_of_structures: int,
        dtype: np.dtype | type = np.float64,
    ) -> float:
        """Estimate the final pooled matrix memory in GiB.

        Possible extensions
        -------------------
        - Estimate peak memory from the largest structure in the dataset.
        - Compare dense and sparse storage estimates.
        - Estimate disk space for cached descriptors.
        """
        # explicit validation
        if not (isinstance(number_of_structures, int) and number_of_structures > 0):
            raise ValueError("number_of_structures must be a positive integer.")

        return (
            number_of_structures
            * self.number_of_features_per_atomic_environment
            * np.dtype(dtype).itemsize
            / (1024**3)
        )

    #### [ 3. Create local SOAP and pool it into crystal descriptors ] ####
    def create_local_soap_descriptors_for_structure(
        self,
        structure: Structure,
    ) -> np.ndarray:
        """Create one SOAP row for every atom in one structure.

        Possible extensions
        -------------------
        - Check translation and rotation invariance numerically.
        - Permute atoms and verify that local rows permute consistently.
        - Time descriptor creation as atom count changes.
        """
        # input validation
        if not isinstance(structure, Structure):
            raise ValueError("structure must be a pymatgen Structure.")

        ase_atoms = AseAtomsAdaptor.get_atoms(structure)  # conversion to ASE Atoms
        local_soap = self._soap.create(ase_atoms)  # evaluation SOAP descriptors

        if self.sparse:
            local_soap = local_soap.toarray()
        else:
            local_soap = np.asarray(local_soap)

        # check dimensions
        expected_shape = (
            len(structure),
            self.number_of_features_per_atomic_environment,
        )

        if local_soap.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {local_soap.shape}"
            )

        return local_soap

    @staticmethod
    def pool_atomic_descriptors_into_structure_descriptor(
        local_atomic_features: np.ndarray,
        pooling_mode: PoolingMode = "mean",
    ) -> np.ndarray:
        """Pool a variable number of atomic rows into one crystal vector.

        Possible extensions
        -------------------
        - Concatenate mean and standard-deviation pooling.
        - Compare mean, sum, maximum, and attention-based pooling.
        - Study what happens when the same unit cell is repeated as a supercell.
        """
        # input validation
        if local_atomic_features.ndim != 2 or local_atomic_features.size == 0:
            raise ValueError(
                "local_atomic_features must be a non-empty 2D NumPy array."
            )

        # pooling
        if pooling_mode == "mean":
            return np.mean(local_atomic_features, axis=0)
        elif pooling_mode == "sum":
            return np.sum(local_atomic_features, axis=0)
        else:
            raise ValueError(
                f"Invalid pooling mode. Must be 'mean' or 'sum', got {pooling_mode}"
            )

    def create_pooled_feature_matrix_for_structures(
        self,
        structures: Sequence[Structure] | pd.Series,
        pooling_mode: PoolingMode = "mean",
        dtype: np.dtype | type = np.float64,  # to manage memory usage
    ) -> np.ndarray:
        """Create one pooled SOAP row for every structure.

        Possible extensions
        -------------------
        - Cache descriptors together with a configuration fingerprint.
        - Add progress reporting and controlled parallel calculation.
        - Process structures in chunks when the matrix is large.
        """
        # explicit validation
        n_structures = len(structures)
        if n_structures == 0:
            raise ValueError("The structure collection is empty.")

        n_features = self.number_of_features_per_atomic_environment

        feature_matrix = np.empty((n_structures, n_features), dtype=dtype)

        for i, s in enumerate(structures):
            local_soap = self.create_local_soap_descriptors_for_structure(s)
            pooled_soap = self.pool_atomic_descriptors_into_structure_descriptor(
                local_soap, pooling_mode
            )
            feature_matrix[i, :] = pooled_soap.astype(dtype)

        if feature_matrix.shape != (n_structures, n_features):
            raise ValueError(
                f"Shape mismatch: expected ({n_structures}, {n_features}), got {feature_matrix.shape}"
            )

        return feature_matrix


#### [ 4. Assemble and check the complete SOAP workflow ] ####
def main() -> None:
    """Run the complete SOAP exercise.

    Possible extensions
    -------------------
    - Save split indices and SOAP configuration for reproducibility.
    - Cache the final features with an unambiguous configuration name.
    - Add timing and memory measurements.
    - Compare mean-pooled SOAP with a simple composition-only baseline.
    """
    print("Loading dataset...")
    dataframe = load_dielectric_dataset()
    print("Dataset loaded.\n")

    print("Extracting species vocabulary...")
    structures = dataframe["structure"]
    sorted_species = collect_sorted_species_list_from_structures(structures)
    print(f"Extracted sorted species vocabulary: {sorted_species}\n")

    print("Creating SOAP featurizer...")
    featurizer = SOAPCrystalStructureFeaturizer(species=sorted_species)
    print(
        f"SOAP feature count: {featurizer.number_of_features_per_atomic_environment}\n"
    )

    print("Memory Safety Check...")
    n_features = featurizer.number_of_features_per_atomic_environment
    n_structures = len(structures)
    print(f"Expected SOAP features per environment: {n_features}")
    memory = featurizer.estimate_dense_feature_matrix_memory_gb(n_structures)
    print(f"Estimated RAM memory for the final matrix: {memory:.4f} GiB\n")

    print("Starting calculations...")
    feature_matrix = featurizer.create_pooled_feature_matrix_for_structures(structures)
    exp_shape = (n_structures, n_features)
    # if feature_matrix.shape != exp_shape, a ValueError is rised
    print(
        f"Final feature matrix shape {feature_matrix.shape} matched the expected shape {exp_shape}\n"
    )


if __name__ == "__main__":
    main()
