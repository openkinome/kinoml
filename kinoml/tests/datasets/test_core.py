"""
Test kinoml.datasets.core
"""


def test_basedatasetprovider():
    from kinoml.datasets.core import BaseDatasetProvider
    from kinoml.core.systems import System
    from kinoml.core.components import MolecularComponent
    from kinoml.core.measurements import BaseMeasurement
    from kinoml.core.conditions import AssayConditions
    from kinoml.features.core import BaseFeaturizer

    conditions = AssayConditions()
    measurements = [
        BaseMeasurement(50, conditions=conditions, system=System([MolecularComponent()])),
        BaseMeasurement(30, conditions=conditions, system=System([MolecularComponent()])),
    ]
    provider = BaseDatasetProvider(measurements=measurements, featurizers=[BaseFeaturizer()])
    assert len(provider.conditions) == 1
    assert next(iter(provider.conditions)) == conditions
