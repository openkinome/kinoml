"""
Test kinoml.core.conditions
"""


def test_assay_conditions():
    from kinoml.core.conditions import AssayConditions

    conditions = AssayConditions()
    assert isinstance(conditions.pH, float)
    assert conditions.pH == 7.0

    assert conditions == AssayConditions(pH=7.0)
    assert conditions != AssayConditions(pH=8.0)
