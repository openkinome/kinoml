"""
Test kinoml.datasets.kinomescan
"""

from kinoml.core.measurements import PercentageDisplacementMeasurement


def test_pkis2():
    from kinoml.datasets.kinomescan.pkis2 import PKIS2DatasetProvider

    provider = PKIS2DatasetProvider.from_source()
    assert len(provider.systems) == 261_870
    assert provider.systems[0].measurement.value == 14.0
