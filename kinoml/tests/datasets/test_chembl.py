"""
Test kinoml.datasets.core
"""


def test_chembl():
    from kinoml.datasets.chembl import ChEMBLDatasetProvider

    # we will use a small subset with 100 entries only, for speed
    chembl = ChEMBLDatasetProvider.from_source(
        "https://github.com/openkinome/kinodata/releases/download/v0.3/activities-chembl29_v0.3.zip",
        uniprot_ids=["P00533"],
        sample=100,
    )
    assert len(chembl) == 100
