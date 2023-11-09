KinoML
==============================
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/openkinome/kinoml/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/openkinome/kinoml/actions/workflows/ci.yml)
[![DOCS](https://github.com/openkinome/kinoml/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/openkinome/kinoml/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/openkinome/KinoML/branch/master/graph/badge.svg)](https://codecov.io/gh/openkinome/KinoML/branch/master)

![GitHub closed pr](https://img.shields.io/github/issues-pr-closed-raw/openkinome/kinoml) 
![GitHub open pr](https://img.shields.io/github/issues-pr-raw/openkinome/kinoml) 
![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/openkinome/kinoml) 
![GitHub open issues](https://img.shields.io/github/issues/openkinome/kinoml)

**KinoML** is a framework for kinase modelling and Machine Learning (ML). It enables users to easily:
1. **Access and download data**: from online data sources, such as ChEMBL or PubChem as well as from their own files, with a focus on data availability and inmutability.
2. **Featurize data**: so that it is ML readeable. KinoML offers a wide variety of featurization schemes, from ligand-only to ligand:kinase complexes.
3. **Run structure-based experiments**: using KinoML's implemented models, with a special focus on reproducibility.

The purpose of KinoML is to help users conduct ML kinase experiments, from data collection to model evaluation. Working examples showcasing how to use KinoML to perform experiments can be found [here.](kinoml/examples/experiments) Note that KinoML has a focus on kinases, but it can be applied to any protein system. For more detailed instructions, working examples, and documentation, please refer to the [Documentation](https://openkinome.org/kinoml/index.html). 


### Notice

Please be aware that this code is work in progress and is not guaranteed to provide the expected results. The API can change at any time without warning.

### Conda/mamba installation

KinoML and its dependencies can be installed via conda/mamba.

```
mamba create -n kinoml --no-default-packages
mamba env update -n kinoml -f https://raw.githubusercontent.com/openkinome/kinoml/master/devtools/conda-envs/test_env.yaml
conda activate kinoml
pip install https://github.com/openkinome/kinoml/archive/master.tar.gz
```

### Usage

Several notebooks providing usage examples can be found in [examples](https://github.com/openkinome/kinoml/tree/master/examples)
including a [getting started notebook](https://github.com/openkinome/kinoml/blob/master/examples/getting_started.ipynb).  
This framework is tightly bound to other repositories:
 - [experiments-binding-affinity](https://github.com/openkinome/experiments-binding-affinity) - for advanced and reproducable ML experiments using KinoML
 - [kinodata](https://github.com/openkinome/kinodata) - ready-to-use kinase-focused datasets from ChEMBL 
### Copyright

Copyright (c) 2019, OpenKinome


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
