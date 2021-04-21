"""
Test kinoml.datasets.core
"""


def test_datasetprovider():
    from kinoml.datasets.core import DatasetProvider
    from kinoml.core.systems import System
    from kinoml.core.components import MolecularComponent
    from kinoml.core.measurements import BaseMeasurement
    from kinoml.core.conditions import AssayConditions

    conditions = AssayConditions()
    measurements = [
        BaseMeasurement(50, conditions=conditions, system=System([MolecularComponent()])),
        BaseMeasurement(30, conditions=conditions, system=System([MolecularComponent()])),
    ]
    provider = DatasetProvider(measurements=measurements)
    assert len(provider.conditions) == 1
    assert next(iter(provider.conditions)) == conditions
