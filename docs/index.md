```{admonition} Warning!
:class: warning

This is module is undergoing heavy development. None of the API calls are final. This software is provided without any guarantees of correctness, you will likely encounter bugs.

If you are interested in this code, please wait for the official release to use it. In the mean time, to stay informed of development progress you are encouraged to:

- Subscribe for new releases (use `Watch> Releases only` on GitHub)
- Check out the [Github repository](https://github.com/openkinome/kinoml).

```

# KinoML

Welcome to the Documentation of KinoML! The documentation is divided into two parts:

* **User guide**: in this section you will learn how to use KinoML to filter and download data from a data base, featurize your kinase data so that it is ML friendly and train and evaluate a ML model on your featurized kinase data. You will also learn about the KinoML object model, and how to access each of these objects. We also provide a detailed examples of how to use every featurizer implemented within KinoML.

* **Experiment tutorials**: this section shows how to use KinoML to  ML structure-based experiments. All experiments are structure-based and they are all end to end, from data collection to model training and evaluation.

    

KinoML falls under the [OpenKinome](https://openkinome.org) initiative, which aims to leverage the increasingly available bioactivity data and scalable computational resources to perform kinase-centric drug design in the context of structure-informed machine learning and free energy calculations. `KinoML` is the main library supporting these efforts.

Do you want to know more about OpenKinome ecosystem? Check its [website](https://openkinome.org).

<!-- Notify Sphinx about the TOC -->

```{toctree}
:caption: User guide
:maxdepth: 1
:hidden:

notebooks/getting_started.nblink
notebooks/kinoml_object_model.nblink
notebooks/OpenEye_structural_featurizer.nblink
notebooks/Schrodinger_structural_featurizer.nblink
```

```{toctree}
:caption: Experiment tutorials
:maxdepth: 1
:hidden:

notebooks/ligand-only-smiles-EGFR.nblink
notebooks/ligand-only-morgan1024-EGFR.nblink
notebooks/kinase-ligand-informed-smiles-sequence-EGFR.nblink
notebooks/kinase-ligand-informed-morgan-composition-EGFR.nblink
```

```{toctree}
:caption: Developers
:maxdepth: 1
:hidden:

API Reference <api/kinoml/index>
```
