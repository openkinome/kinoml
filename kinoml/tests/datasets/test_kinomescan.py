"""
Test kinoml.datasets.kinomescan
"""

import pytest


def test_kinomescan_mapper():
    from kinoml.datasets.kinomescan.utils import KINOMEScanMapper

    mapper = KINOMEScanMapper()
    assert mapper.sequence_for_name("ABL2") == mapper.sequence_for_accession("NP_005149.4")


@pytest.mark.slow
def test_pkis2():
    from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider

    provider = PKIS2DatasetProvider.from_source()
    assert len(provider.measurements) == 261_870
    assert (provider.measurements[0].values == 14.0).all()


def test_access_by_index_roundtrip():
    """
    Check notes in `kinoml.dataset.kinomescan.pkis2.PKIS2DatasetProvider.from_source()`
    """
    raise NotImplementedError("This test is pending and important!")
