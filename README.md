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

**KinoML** is a modular and extensible framework for machine learning (ML) in small molecule drug discovery with a special focus on kinases. It enables users to easily:
1. **Access and download data**: from online data sources, such as ChEMBL or PubChem as well as from their own files, with a focus on data availability and inmutability.
2. **Featurize data**: so that it is ML readeable. KinoML offers a wide variety of featurization schemes, from ligand-only to ligand:kinase complexes.
3. **Run structure-based experiments**: using KinoML's implemented models, with a special focus on reproducibility.


The purpose of KinoML is to help users conduct ML kinase experiments, from data collection to model evaluation. Tutorials on how to use KinoML as well as working examples showcasing how to use KinoML to perform experiments end-to-end can be found [here.](https://github.com/raquellrios/kinoml/tree/master/tutorials) Note that despite KinoML's focus being on kinases, it can be applied to any protein system. For more detailed instructions, please refer to the [Documentation](https://openkinome.org/kinoml/index.html). 



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

The tutorials folder is divided into two parts:

1. [**Getting started**](https://github.com/raquellrios/kinoml/tree/master/tutorials/getting_started): the notebooks in this folder aim to give the user an understanding of how to use KinoML to: (1) **access and download** data, (2) **featurize** data, and (3) **run a** (simple) **ML model** on the featurized data obtained with KinoML to predict ligand binding affinity. Additionally, this folder contains notebooks that explain the **KinoML object model** and how to access the different objects, as well as notebooks **showcasing all the different featurizers** implemented within KinoML and how to use each of them.

2. [**Experiments**](https://github.com/raquellrios/kinoml/tree/master/tutorials/experiments): this folder contains four individual structure-based experiments to predict ligand binding affinity. All experiments use KinoML to obtain the data, featurize it and train and evaluate a ML model implemented within the`kinoml.ml` class. The purpose of these experiments is to display usage examples of KinoML to conduct end-to-end structure-based kinases experiments.


⚠️ You will need a valid OpenEye License for the featurizers of the tutorials to work. For the Schrodinger featurizers tutorial you will also need a Schrodinger License!


For users interested in more KinoML usage examples, they can checkout other repositories under the initative [OpenKinome](https://github.com/openkinome/). Particularly, other two repositories that may be of interest are:


- [kinodata](https://github.com/openkinome/kinodata): repository with ready-to-use kinase-focused datasets from ChEMBL, as well as tutorials explaining how to process kinase data for ML applications. 
- [experiments-binding-affinity](https://github.com/openkinome/experiments-binding-affinity): more advanced and reproducible ML experiments using KinoML.



Copyright (c) 2019, OpenKinome


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
