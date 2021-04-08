"""
Test kinoml.datasets.core
"""


def test_chembl():
    from kinoml.datasets.chembl import ChEMBLDatasetProvider

    chembl = ChEMBLDatasetProvider.from_source(
        "https://github.com/openkinome/kinodata/releases/download/v0.2/activities-chembl28-sample100_v0.2.zip"
    )
    assert len(chembl) == 100