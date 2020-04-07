"""
Test kinoml.core.measurements
"""


def test_measurements():
    from kinoml.core.measurements import BaseMeasurement, PercentageDisplacementMeasurement
    from kinoml.core.conditions import AssayConditions
    from kinoml.core.components import MolecularComponent

    conditions = AssayConditions()
    components = [MolecularComponent()]
    measurement = BaseMeasurement(50, conditions=conditions, components=components)
    assert isinstance(measurement, BaseMeasurement)
    assert measurement == BaseMeasurement(50, conditions=conditions, components=components)
    assert measurement != BaseMeasurement(10, conditions=conditions, components=components)
