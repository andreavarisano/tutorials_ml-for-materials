"""Step 02a: create fixed-size crystal descriptors using SOAP.
Complete each function and method by replacing raise NotImplementedError.

Learning goals
--------------
1. Collect the chemical species present in a set of crystal structures.
2. Configure a SOAP descriptor and understand its main hyperparameters.
3. Convert each pymatgen Structure into local, atom-centred SOAP descriptors.
4. Pool a variable number of local atomic descriptors into one fixed-size
   vector per crystal.
5. Check descriptor dimension and memory cost before processing the full dataset.


A practical step-by-step workflow: construct your SOAP descriptor with DScribe
---------------------------------------
Two separate operations: 
    first configure a SOAP calculator -> use that calculator to evaluate structures.

1. Collect the species vocabulary
   SOAP must be told all chemical elements it may encounter:
       species = collect_sorted_species_list_from_structures(structures)
2. Construct the DScribe SOAP calculator
   At this stage you should choose the hyperparameters that define the descriptor (and keep them fixed for every structure).
   soap = SOAP( species=species, periodic=True, r_cut=5.0, n_max=6, l_max=4,
                average="off",   sparse=False ,)
     r_cut defines the local neighbourhood radius, 
     n_max controls the radial basis size,
     l_max controls the angular basis.      
3. Convert one pymatgen structure to ASE
   DScribe expects an ASE Atoms object, while Matminer supplies pymatgen.Structure objects. Convert:
       ase_atoms = AseAtomsAdaptor.get_atoms(structure)
4. Evaluate local SOAP descriptors
   With average="off", calling create without specifying centres evaluates one local environment per atom:
       local_soap = soap.create(ase_atoms)
   Its expected shape is (number_of_atoms, number_of_soap_features).
5. Convert local descriptors into one structure descriptor
   Pool over the atomic axis, for example; expected shape is (number_of_soap_features,).
6. Repeat for all structures and stack the vectors
   Preserve the original structure order so rows remain aligned with targets and split indices:
       feature_matrix = np.stack(structure_descriptors, axis=0)
   Expected shape is (number_of_structures, number_of_soap_features).
   
7. Inspect size before processing the full dataset
   Ask SOAP for its feature count and estimate the dense matrix memory before
   running all structures. First test the complete path on a few small crystals.
The SOAPCrystalStructureFeaturizer below packages these steps, but each method keeps one responsibility visible.


A tiny bit of knowledge: local and global descriptors
-----------------------------------------------------
SOAP describes the environment around one atom. Therefore, for a structure
containing N atoms, local SOAP initially returns a matrix with shape:
    (number_of_atoms, number_of_soap_features)

Different structures contain different numbers of atoms, so these matrices cannot be stacked directly.
For a standard ML algorithm we need one fixed-size vector per structure. 
A simple solution is to pool over the atomic dimension:
    mean pooling: local_soap.mean(axis=0)
    sum pooling:  local_soap.sum(axis=0)

The resulting structure descriptor has shape:  (number_of_soap_features,)
Think about the physical difference: sum pooling retains information about the number of atoms in the selected cell, whereas mean pooling removes it.
Note: DScribe can also perform averaging internally through the SOAP constructor's average argument:
    average="off"    # one local SOAP descriptor per selected centre
    average="outer"  # average the SOAP power spectra over the centres
    average="inner"  # average coefficients before constructing the power spectrum
The "outer" mode is the built-in option most closely related to mean-pooling local SOAP vectors. The "inner" mode averages at an earlier mathematical stage.

In this exercise we start by average="off" and perform pooling explicitly. This makes the local descriptor shape and atomic axis visible, and lets us compare mean and sum pooling. 

A tiny bit of knowledge: dataclass, __init__, and __post_init__
---------------------------------------------------------------
SOAPCrystalStructureFeaturizer is decorated with @dataclass.
The dataclass decorator automatically creates an __init__ method from the annotated fields,
which are : species, r_cut, n_max, l_max, periodic, sparse
This is why no __init__ method is explicitly written below.
After the generated __init__ assigns those fields, it automatically calls __post_init__ - which is explicitly defined below.
__post_init__ role here is to validate the values received by the generated constructor and then create the internal DScribe SOAP object.
The double underscores indicate a Python special method. 


A tiny bit of knowledge: descriptor dimension and memory
--------------------------------------------------------
It's good practice to have a rough idea of the final feature matrix size before computing descriptors for every structure;
this because memory can often be a limiting factor (or a bottleneck) in ML workflows. 
Furthermore, up until now we have used float64 as the default data type for our features, but is this precision really necessary? Could we use float32 instead?

Proposal : a quick test, let's estimate the size of the final dense matrix before computing descriptors for every structure.
Before computing descriptors for every structure, count the species and create a SOAP object with the chosen hyperparameters.
Estimated size : number_of_structures * number_of_soap_features * bytes_per_number
Remember that this estimates only the pooled matrix. Temporary local SOAP arrays and library overhead require additional memory.

Questions for the curious mind :
- How does the number of features change when another species is added?
- Is float64 necessary, or would float32 be sufficient?
- Would a restricted chemical subset, compressed SOAP, or PCA be useful?
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from step01_load_dataset import load_dielectric_dataset


# ---------------------------------------------------------------------------
# Original exploratory stub
# ---------------------------------------------------------------------------
# This is the initial one-off workflow. The exercise is to reorganize these
# operations into the reusable functions and class defined below.
#
# from matminer.datasets import load_dataset
#
# df = load_dataset("matbench_dielectric")
# df_structures = df["structure"]
# df_targets = df["n"]
#
# # Collect every element appearing anywhere in the dataset.
# species = sorted({
#     str(element)
#     for structure in df_structures
#     for element in structure.composition.elements
# })
#
# print(f"Number of species: {len(species)}")
# print(species)
#
# soap = SOAP(
#     species=species,
#     periodic=True,
#     r_cut=5.0,
#     n_max=6,
#     l_max=4,
#     sparse=False,
# )


PoolingMode = Literal["mean", "sum"]


#### [ 1. Collect the chemical species present in the dataset ] ####
def collect_sorted_species_list_from_structures(
    structures: Sequence[Structure] | pd.Series,
) -> list[str]:
    """Return every chemical species in the structures exactly once.
    Requirements
    ------------
    - Accept the pandas Series returned by dataframe["structure"] (or another ordered sequence of pymatgen Structure objects).
    - Iterate through structure.composition.elements.
    - Convert each element to its string symbol.
    - Remove repeated symbols, including repetitions across structures.
    - Sort the symbols so the descriptor configuration is deterministic.
    - Reject an empty structure collection with an informative error.

    Beware / remember
    -----------------
    - The species list will be an input to the soap constructor function. In the standard implementation,
      the ordering of species affects the ordering of SOAP features. 
      Moreover, a Python set  removes duplicates but does not by itself define the final ordering.
    - Tips: This function should define only the vocabulary: do not create SOAP or calculate descriptors here.
      Where possible, do not overload a single function with too many goals.

    Advice
    ------
    First test the function on two or three structures for which you can list the expected elements by hand.
    """
    # input validation
    if len(structures) == 0:
        raise ValueError("The structure collection is empty.")

    return sorted({
        str(element)
        for structure in structures
        for element in structure.composition.elements
    })

#### [ 2. Configure the SOAP crystal-structure featurizer ] ####
@dataclass
class SOAPCrystalStructureFeaturizer:
    """Configuration and operations needed to create crystal SOAP features.

    Beware / remember
    -----------------
    A class is a suggested interface, not a requirement. Another organization
    is acceptable if species collection, SOAP configuration, descriptor
    creation and pooling remain clearly separated and documented.

    Advice
    ------
    Keep the configuration values accessible on the object. They are part of
    the scientific definition of the descriptor and should eventually be saved
    with any trained model.

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

        Requirements
        ------------
        - Check that species is non-empty.
        - Check that species contains no duplicate or empty symbols.
        - Preserve a deterministic species ordering.
        - Check that r_cut is finite and strictly positive.
        - Check that n_max is a strictly positive integer.
        - Check that l_max is a non-negative integer.
        - Check that periodic and sparse are booleans.
        - Create dscribe.descriptors.SOAP using the validated values.
        - Assign the resulting descriptor to self._soap.

        Beware / remember
        -----------------
        @dataclass generates __init__ automatically from the annotated fields.
        That generated method calls __post_init__ after assigning the public
        fields. Do not call __post_init__ manually.

        The field _soap has init=False, so users cannot pass it to the generated
        constructor. It must be created here after validation. Also remember
        that type annotations document types but do not enforce them at runtime.

        Advice
        ------
        Validate all inputs before creating self._soap so an invalid object is
        never left partially initialized. Give errors that identify the invalid
        parameter and value.

        Possible extensions
        -------------------
        - Validate symbols using a periodic-table utility.
        - Add DScribe compression or species-weighting settings.
        - Add a method that compares two featurizer configurations.
        """
        # input validation
        if len(self.species) == 0:
            raise ValueError("The species collection is empty.")
        if len(self.species) != len(set(self.species)):
            raise ValueError("The species collection contains duplicates.")
        if self.species != sorted(self.species):
            raise ValueError("The species collection is not sorted.")

        if not (isinstance(self.r_cut, (int, float)) and np.isfinite(self.r_cut) and self.r_cut > 0):
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
            average="off"
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

        Requirements
        ------------
        - Copy species into a plain list so the featurizer owns its settings.
        - Forward all settings to cls using explicit keyword arguments.
        - Do not inspect structures or collect species in this method.

        Beware / remember
        -----------------
        A @classmethod receives the class as cls, not an existing instance as
        self. Calling cls(...) invokes the dataclass-generated __init__, which
        then automatically invokes __post_init__.

        Advice
        ------
        Keep this named constructor small. Validation belongs in __post_init__
        so direct construction and classmethod construction behave identically.

        Possible extensions
        -------------------
        Add a create_featurizer_from_configuration_dictionary classmethod for
        recreating an exactly logged descriptor.
        """
        species_copy = list(species) # copy

        return cls(
            species=species_copy,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            periodic=periodic,
            sparse=sparse,
        )

    @property
    def number_of_features_per_atomic_environment(self) -> int:
        """Return the number of values in one local SOAP descriptor.

        Requirements
        ------------
        - Ask self._soap for its feature count.
        - Return a positive Python integer.

        Beware / remember
        -----------------
        @property makes this value accessible without call parentheses. Do not
        copy DScribe's dimension formula into this property: the library is the
        authoritative source for its configured descriptor size.

        Advice
        ------
        Print this value before calculating any descriptors and compare it
        after changing one SOAP hyperparameter at a time.

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

        Requirements
        ------------
        - Check that number_of_structures is a positive integer.
        - Use numpy.dtype(dtype).itemsize to obtain bytes per value.
        - Use number_of_features_per_atomic_environment for the column count.
        - Convert bytes to GiB using 1024**3.

        Beware / remember
        -----------------
        This estimates only the final pooled matrix. Local per-atom descriptors,
        intermediate arrays, Python objects, and library overhead require extra
        memory. GiB uses 1024**3 bytes, whereas GB conventionally uses 1000**3.

        Advice
        ------
        Estimate both float64 and float32 storage before choosing a dtype. Do
        not compute the full dataset merely to discover that it does not fit.

        Possible extensions
        -------------------
        - Estimate peak memory from the largest structure in the dataset.
        - Compare dense and sparse storage estimates.
        - Estimate disk space for cached descriptors.
        """
        # input validation
        if not (isinstance(number_of_structures, int) and number_of_structures > 0):
            raise ValueError("number_of_structures must be a positive integer.")

        return number_of_structures * self.number_of_features_per_atomic_environment * np.dtype(dtype).itemsize / (1024**3)


    #### [ 3. Create local SOAP and pool it into crystal descriptors ] ####
    def create_local_soap_descriptors_for_structure(
        self,
        structure: Structure,
    ) -> np.ndarray:
        """Create one SOAP row for every atom in one structure.

        Requirements
        ------------
        - Check that structure is a pymatgen Structure.
        - Convert it to ASE Atoms using AseAtomsAdaptor.
        - Preserve lattice, periodic boundaries, species, and coordinates.
        - Pass the ASE object to self._soap.create.
        - For the dense exercise, return an array shaped:
              (number_of_atoms, number_of_features_per_atomic_environment)
        - Check both dimensions before returning.

        Beware / remember
        -----------------
        Losing the periodic cell or periodic-boundary flags changes the physical
        environments being described. If sparse=True is supported, explicitly
        handle its output type rather than assuming a dense NumPy array.

        Advice
        ------
        Begin with one small structure. Print the pymatgen structure, converted
        ASE object, output dtype, and output shape before processing more data.

        Possible extensions
        -------------------
        - Check translation and rotation invariance numerically.
        - Permute atoms and verify that local rows permute consistently.
        - Time descriptor creation as atom count changes.
        """
        # input validation
        if not (isinstance(structure, Structure)):
            raise ValueError("structure must be a pymatgen Structure.")

        ase_atoms = AseAtomsAdaptor.get_atoms(structure) # conversion to ASE Atoms
        local_soap = self._soap.create(ase_atoms) # evaluation SOAP descriptors

        if self.sparse:
            local_soap = local_soap.toarray()
        else:
            local_soap = np.asarray(local_soap)

        # check dimensions
        expected_shape = (len(structure), self.number_of_features_per_atomic_environment)
        
        if local_soap.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {local_soap.shape}")

        return local_soap

    @staticmethod
    def pool_atomic_descriptors_into_structure_descriptor(
        local_atomic_features: np.ndarray,
        pooling_mode: PoolingMode = "mean",
    ) -> np.ndarray:
        """Pool a variable number of atomic rows into one crystal vector.

        Requirements
        ------------
        - Require a non-empty two-dimensional array.
        - Accept only "mean" and "sum" as pooling modes.
        - Pool over axis 0, the atomic axis.
        - Return a one-dimensional vector with one value per SOAP feature.
        - Raise an informative error for invalid arrays or modes.

        Beware / remember
        -----------------
        Pooling over axis 1 would collapse the SOAP features instead of the
        atoms. Mean and sum pooling are both permutation invariant, but sum
        depends on the number of atoms in the chosen cell while mean does not.

        Advice
        ------
        Test pooling with a tiny hand-written matrix whose mean and sum you can
        calculate manually. Check the returned shape as well as its values.

        Possible extensions
        -------------------
        - Concatenate mean and standard-deviation pooling.
        - Compare mean, sum, maximum, and attention-based pooling.
        - Study what happens when the same unit cell is repeated as a supercell.
        """
        # input validation
        if not isinstance(local_atomic_features, np.ndarray):
            raise ValueError("local_atomic_features must be a NumPy array.")
        if local_atomic_features.ndim != 2:
            raise ValueError(f"local_descriptors must be 2D, got {local_atomic_features.ndim}D")
        if local_atomic_features.size == 0:
            raise ValueError("The local_atomic_features array is empty.")

        # pooling
        if pooling_mode == "mean":
            return np.mean(local_atomic_features, axis=0)
        elif pooling_mode == "sum":
            return np.sum(local_atomic_features, axis=0)
        else:
            raise ValueError(f"Invalid pooling mode. Must be 'mean' or 'sum', got {pooling_mode}")

    def create_pooled_feature_matrix_for_structures(
        self,
        structures: Sequence[Structure] | pd.Series,
        pooling_mode: PoolingMode = "mean",
    ) -> np.ndarray:
        """Create one pooled SOAP row for every structure.

        Requirements
        ------------
        - Reject an empty structure collection.
        - Preserve the original structure order.
        - Create local SOAP descriptors for each structure.
        - Pool each local matrix using the selected pooling mode.
        - Stack the pooled vectors into a matrix shaped:
              (number_of_structures, number_of_features_per_atomic_environment)
        - Verify the final shape.
        - Do not preprocess or scale features inside this method; that belongs to Step 04.

        Beware / remember
        -----------------
        The row order must remain aligned with targets and split indices. A
        silent reordering would train on incorrect structure-target pairs.
        Estimate memory before using this method on the complete dataset.

        Advice
        ------
        Run the full path on a small slice first. For a dense matrix with known
        width, consider preallocating the final array instead of storing every
        local matrix simultaneously.

        Possible extensions
        -------------------
        - Cache descriptors together with a configuration fingerprint.
        - Add progress reporting and controlled parallel calculation.
        - Process structures in chunks when the matrix is large.
        """
        # input validation
        n_structures = len(structures)
        if n_structures == 0:
            raise ValueError("The structure collection is empty.")

        n_features = self.number_of_features_per_atomic_environment

        feature_matrix = np.empty((n_structures, n_features))

        for i, s in enumerate(structures):
            local_soap = self.create_local_soap_descriptors_for_structure(s)
            pooled_soap = self.pool_atomic_descriptors_into_structure_descriptor(local_soap, pooling_mode)
            feature_matrix[i, :] = pooled_soap

        if feature_matrix.shape != (n_structures, n_features):
            raise ValueError(f"Shape mismatch: expected ({n_structures}, {n_features}), got {feature_matrix.shape}")

        return feature_matrix


#### [ 4. Assemble and check the complete SOAP workflow ] ####
def main() -> None:
    """Run the complete SOAP exercise.

    Requirements
    ------------
    - Load structures using Step 01.
    - Collect and print the sorted species vocabulary.
    - Create the SOAP featurizer from that vocabulary.
    - Print the SOAP feature count and memory estimate before calculation.
    - Create the pooled structure feature matrix.
    - Print the final feature-matrix shape as a consistency check.

    Beware / remember
    -----------------
    Do not compute the full descriptor matrix before checking its dimension and
    estimated memory. Keep structure rows, targets, and split indices aligned.

    Advice
    ------
    Make the workflow run successfully on a handful of structures before using
    the complete dataset. Print shapes at every boundary.

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
    structures = dataframe['structure']
    sorted_species = collect_sorted_species_list_from_structures(structures)
    print(f"Extracted sorted species vocabulary: {sorted_species}\n")

    print("Creating SOAP featurizer...")
    featurizer = SOAPCrystalStructureFeaturizer(species=sorted_species)
    print(f"SOAP feature count: {featurizer.number_of_features_per_atomic_environment}\n")

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
    print(f"Final feature matrix shape {feature_matrix.shape} matched the expected shape {exp_shape}\n")


if __name__ == "__main__":
    main()
