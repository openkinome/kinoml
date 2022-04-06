# Installation

KinoML and its dependencies can be installed via conda/mamba.:

```
mamba create -n kinoml --no-default-packages
mamba env update -n kinoml -f https://raw.githubusercontent.com/openkinome/kinoml/master/devtools/conda-envs/test_env.yaml
conda activate kinoml
pip install https://github.com/openkinome/kinoml/archive/master.tar.gz
```

Note this has only been tested in Linux and MacOS. In the future, a `conda` package `kinoml` will be provided, but right now we are in a very early stage of development.
