"""
Test kinoml.core.sequences and derived objects
"""

from ...core.sequences import Biosequence
from ...core.proteins import AminoAcidSequence


def test_biosequence_mutation():
    s = Biosequence("ATCGTHCTCH")
    s.substitute("C3P")
    assert s.sequence == "ATPGTHCTCH"
    s.delete(2, 5)
    assert s.sequence == "AHCTCH"
    s.delete(2, 5, insert="AA")
    assert s.sequence == "AAAH"
    s.insert(5, "T")
    assert s.sequence == "AAAHT"


def test_aminoacidsequence_fetching():
    s1 = AminoAcidSequence(uniprot_id="P00519-1")
    s2 = AminoAcidSequence(ncbi_id="NP_005148.2")
    assert s1.sequence == s2.sequence
