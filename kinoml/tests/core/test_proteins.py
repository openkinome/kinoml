"""
Test kinoml.core.proteins
"""


def test_proteins():
    from kinoml.core.components import BaseProtein
    from kinoml.core.proteins import AminoAcidSequence

    sequence = AminoAcidSequence("AAAAAAAAA", name="AAA")
    assert isinstance(sequence, BaseProtein)
    assert sequence == AminoAcidSequence("AAAAAAAAA", name="AAA")
    assert sequence == AminoAcidSequence("AAAAAAAAA", name="A")
