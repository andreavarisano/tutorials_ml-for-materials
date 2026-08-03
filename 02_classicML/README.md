1] Matbench Dataset
 Matbench already provides "pre-packaged" dataset, which could be a useful start. See https://hackingmaterials.lbl.gov/matminer/dataset_summary.html
 I would focus in particular on the datasets: 
    matbench_dielectric		Description= "Matbench v0.1 test dataset for predicting refractive index from structure  Adapted from Materials Projects Database"
    				Reference= Petousis et al. High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).
				Material Number = "4764"
    matbench_expt_gap		Description="Matbench v0.1 test dataset for predicting experimental band gap from composition alone"
				Material Number = "4604"
 - How to load them:  
   See https://hackingmaterials.lbl.gov/matminer/index.html#access-ready-made-datasets-in-one-line for more details.

2] Feature-construction exercise sequence

01_Feature_Construction/

- step01_load_dataset.py
  Load and inspect matbench_dielectric and create reproducible split indices.

- step02a_create_soap.py
  Construct local SOAP descriptors and pool them into crystal-level vectors.

- step02b_create_composition_features.py
  Construct Stoichiometry + Magpie composition descriptors and preserve their
  feature names.

- step03_concatenate_features.py
  Validate row alignment and concatenate SOAP and composition columns.

- step04_preprocessing.py
  Fit target and feature scalers using training data only, then reuse them for
  validation and test data.

Steps 02a and 02b are parallel feature-construction branches. Both feed Step 03.
The filenames are valid Python module names and can be imported normally.
