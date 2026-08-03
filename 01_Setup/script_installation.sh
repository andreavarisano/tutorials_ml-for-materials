python3 -m venv venv_matbench
source venv_matbench/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install dscribe torch matminer 
#Note: after version 0.6, "matbench @ git+https://github.com/materialsproject/matbench.git" was moved into matminer
#The git matbench repo is a bit abandoned (last commit 2 years ago), so we use the other

