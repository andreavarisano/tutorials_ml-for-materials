# Machine Learning for Materials Tutorials

Hands-on exercises for learning machine learning with materials-science data.
The repository starts with structure descriptors and classical regression, then
progresses toward graph neural networks and optical-property prediction.

The exercises are written for students who already know basic Python and
introductory PyTorch concepts such as tensors, automatic differentiation,
forward passes, and optimization.

## Goals

- Work with real crystal structures and materials datasets.
- Build SOAP and composition-based descriptors.
- Create reproducible data splits and avoid preprocessing leakage.
- Compare composition-only, structure-only, and combined representations.
- Prepare for invariant graph neural networks and, later, dielectric-spectrum
  prediction.

## Repository structure

```text
01_Setup/
    Environment setup script.

02_classicML/
    Classical machine-learning exercises using Matbench, SOAP, and Matminer
    to construct descriptors, followed by simple neural-network models in
    PyTorch.

    Steps 02a and 02b are parallel feature-construction exercises. Their
    outputs are combined in Step 03 and preprocessed in Step 04.


    01_Feature_Construction/
        step01_load_dataset.py
        step02a_create_soap.py
        step02b_create_composition_features.py
        step03_concatenate_features.py
        step04_preprocessing.py

03_STUB_GNN/
    Initial blueprint for the graph-neural-network section.
```


Most exercise functions intentionally raise `NotImplementedError`. Students
should complete them by following the requirements, warnings, and suggestions
provided in each file.

## Setup

Python 3.10 or newer is recommended.


### Cloning the repo
Cloning the original repository is sufficient for reading and running the
exercises.
```bash
git clone https://github.com/lorenzovarrassi/tutorials_ml-for-materials.git
cd tutorials_ml-for-materials

bash 01_Setup/script_installation.sh
source venv_matbench/bin/activate
```

The setup installs the main dependencies, including PyTorch, Matminer, and
DScribe. The virtual environment is excluded from version control.

### Working from a fork

 Students who want to complete the exercises and push their work via git should first create a personal fork:

1. Click **Fork** on the GitHub repository page.
2. Clone the fork, replacing `YOUR_USERNAME` below:

   ~~~bash
   git clone https://github.com/YOUR_USERNAME/tutorials_ml-for-materials.git
   cd tutorials_ml-for-materials
   ~~~

3. Create a branch for the solutions:

   ~~~bash
   git switch -c student-solutions
   ~~~

4. Commit and push changes to the personal fork:

   ~~~bash
   git add .
   git commit -m "Complete feature-construction exercises"
   git push -u origin student-solutions
   ~~~

Optionally, keep a reference to the original repository for future updates:

~~~bash
git remote add upstream https://github.com/lorenzovarrassi/tutorials_ml-for-materials.git
git fetch upstream
~~~

## Dataset

The first exercises use `matbench_dielectric`, a Matbench dataset containing
crystal structures and refractive-index targets. It provides a small but real
materials problem connected to the later study of optical properties.

Dataset documentation:  
<https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>

## Suggested workflow

1. Read the introduction and requirements in each exercise.
2. Test every function on a small number of structures first.
3. Check array shapes and preserve material row order.
4. Fit scalers and other preprocessing objects using training data only.
5. Record descriptor settings, split seeds, and model hyperparameters.

## Project status

This repository is under active development. The current material focuses on
dataset preparation and feature construction; graph-neural-network and optical
spectrum exercises will be added progressively.

## License

This project is distributed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for details.
