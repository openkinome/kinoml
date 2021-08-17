KinoML
==============================
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/openkinome/kinoml/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/openkinome/kinoml/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/openkinome/KinoML/branch/master/graph/badge.svg)](https://codecov.io/gh/openkinome/KinoML/branch/master)

![GitHub closed pr](https://img.shields.io/github/issues-pr-closed-raw/openkinome/kinoml) 
![GitHub open pr](https://img.shields.io/github/issues-pr-raw/openkinome/kinoml) 
![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/openkinome/kinoml) 
![GitHub open issues](https://img.shields.io/github/issues/openkinome/kinoml)

Machine Learning for kinase modeling. 

### Notice

Please be aware that this code is work in progress and is not guaranteed to provide the expected results. The API can change at any time without warning.

### Installation

KinoML and its dependencies can be installed via conda/mamba.

```
git clone https://github.com/openkinome/kinoml.git
cd kinoml
mamba env create -n kinoml -f devtools/conda-envs/test_env.yaml
conda activate kinoml
pip install .
```

### Usage

Several notebooks providing usage examples can be found in [examples](https://github.com/openkinome/kinoml/tree/master/examples).  
Also, this framework used by several repositories, which may give additional insights:
 - [experiments-binding-affinity](https://github.com/openkinome/experiments-binding-affinity)
 - [kinase-conformational-modeling](https://github.com/openkinome/kinase-conformational-modeling)
 - [study-abl-resistance](https://github.com/openkinome/study-abl-resistance)
 - [study-ntrk-resistance](https://github.com/openkinome/study-ntrk-resistance)

### Copyright

Copyright (c) 2019, OpenKinome


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
