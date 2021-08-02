KinoML
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/KinoML.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/KinoML)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/KinoML/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/KinoML/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/KinoML/branch/master)

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
