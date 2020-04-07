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
    components_1 = [MolecularComponent()]
    components_2 = [MolecularComponent()]
    systems = [
        System(
            components=components_1,
            measurement=BaseMeasurement(50, conditions=conditions, components=components_1),
        ),
        System(
            components=components_2,
            measurement=BaseMeasurement(30, conditions=conditions, components=components_2),
        ),
    ]
    provider = BaseDatasetProvider(systems=systems, featurizers=[BaseFeaturizer()])
    assert len(provider.assay_conditions) == 1
    assert next(iter(provider.assay_conditions)) == conditions
