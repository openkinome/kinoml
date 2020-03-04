# Base concepts in our API


## DatasetProvider


This object is responsible for:

- Taking raw data as provided by source/calculation/experiment
- Featurizing the input chemical information to obtain numerical values
- Transforming (featurizing) the known measurements to compatible magnitudes
- Generating downstream Dataset-like objects natively compatible with the expected framework (tensorflow, pytorch, scikit-learn, etc).

The process of going from chemical-like information to numerical information is called `featurization`, and is performed by `featurizers`. Most featurizers will work in a one-by-one basis: they only need one input to provide the output. Some, however, could need contextual information from the dataset to perform that operation (averaging measurements, caching already existing structures, etc). As a result, they all can take an additional optional keyword argument holding the whole dataset.

```python
def featurizer(data_point_to_featurize, dataset=None):
    pass
```

    Chemical data -------[featurization]----> Numerical n-dimensional arrays ---> Framework-native object


In this step, we might also perform filtering: discard non-desired/compatible data points. This is done with `Filter` objects. Filtering alters the shape of the data because it will always remove data points following some criterion.

## Dataset


This object contains numerical data only. They can be post-processed with `Transformer` objects native to the framework in use.

## MolecularSystem

This object is central to DatasetProvider, because it describes the object model that will hold all the chemical information we can infer from the raw data.

For example:

- Smiles -> Ligand(MolecularSystem)
- Protein Sequences -> Protein(MolecularSystem)
- PDB -> Complex(MolecularSystem)

They need to provide most of the information featurizers will expect. To this effect, MolecularSystem will provide flags that report what type of info is available for featurizers to check for compatibility.