"""
Test kinoml.core.measurements
"""


def test_measurements():
    from kinoml.core.measurements import (
        BaseMeasurement,
        PercentageDisplacementMeasurement,
    )
    from kinoml.core.conditions import AssayConditions
    from kinoml.core.components import MolecularComponent
    from kinoml.core.systems import System

    conditions = AssayConditions()
    system = System([MolecularComponent()])
    measurement = BaseMeasurement(50, conditions=conditions, system=system)
    assert isinstance(measurement, BaseMeasurement)
    assert measurement == BaseMeasurement(50, conditions=conditions, system=system)
    assert measurement != BaseMeasurement(10, conditions=conditions, system=system)
