"""
Test kinoml.datasets.kinomescan
"""


def test_kinomescan_mapper():
    from kinoml.datasets.kinomescan.utils import KINOMEScanMapper

    mapper = KINOMEScanMapper()
    assert mapper.sequence_for_name("ABL2") == mapper.sequence_for_accession("NP_005149.4")


def test_pkis2():
    from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider

    provider = PKIS2DatasetProvider.from_source()
    assert len(provider.systems) == 261_870
    assert provider.systems[0].measurement.value == 14.0
