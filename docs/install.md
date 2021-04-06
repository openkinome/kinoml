# Installation

Assuming you have a local copy of the KinoML repository, `cd` into the repo and use `conda` to create an environment:

1. `conda env create -n kinoml -f devtools/conda-envs/test_env.yaml`
2. `conda activate kinoml`
3. `pip install .`

Note this has only been tested in Linux and MacOS. In the future, a `conda` package `kinoml` will be provided, but right now we are in a very early stage of development.
