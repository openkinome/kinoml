"""
Test kinoml.datasets.core
"""


def test_chembl():
    from kinoml.datasets.chembl import ChEMBLDatasetProvider

    # we will use a small subset with 100 entries only, for speed
    chembl = ChEMBLDatasetProvider.from_source(
        "https://github.com/openkinome/kinodata/releases/download/v0.2/activities-chembl28-sample100_v0.2.zip"
    )
    assert len(chembl) == 100