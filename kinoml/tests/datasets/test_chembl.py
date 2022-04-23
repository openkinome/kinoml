"""
Test kinoml.datasets.core
"""


def test_chembl():
    from kinoml.core.proteins import Protein, KLIFSKinase
    from kinoml.datasets.chembl import ChEMBLDatasetProvider

    # we will use a small subset with 100 entries only, for speed
    chembl = ChEMBLDatasetProvider.from_source(
        "https://github.com/openkinome/kinodata/releases/download/v0.3/activities-chembl29_v0.3.zip",
        uniprot_ids=["P00533"],
        sample=100,
        protein_type="Protein",
        toolkit="OpenEye",
    )
    assert len(chembl) == 100
    assert isinstance(chembl.systems[0].protein, Protein)
    assert chembl.systems[0].protein.toolkit == "OpenEye"

    chembl = ChEMBLDatasetProvider.from_source(
        "https://github.com/openkinome/kinodata/releases/download/v0.3/activities-chembl29_v0.3.zip",
        uniprot_ids=["P00533"],
        sample=100,
        protein_type="KLIFSKinase",
        toolkit="MDAnalysis",
    )
    assert len(chembl) == 100
    assert isinstance(chembl.systems[0].protein, KLIFSKinase)
    assert chembl.systems[0].protein.toolkit == "MDAnalysis"
