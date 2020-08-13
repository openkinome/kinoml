"""
Test kinoml.core.sequences and derived objects
"""

from ...core.sequences import Biosequence
from ...core.proteins import AminoAcidSequence


def test_biosequence_mutation():
    s = Biosequence("ATCGTHCTCH")
    assert s.mutate("C3P") == "ATPGTHCTCH"
    assert s.mutate("T2-T5del") == "ATTHCTCH"
    assert s.mutate("5Tins") == "ATCGTTHCTCH"
    assert s.mutate("A1T", "T2A") == "TACGTHCTCH"
    assert s.mutate("A1T", "T2A", "3Tins") == "TACTGTHCTCH"
    assert s.mutate("A1T", "T2A", "T2-G4del") == "TAGTHCTCH"


def test_biosequence_cut():
    s = Biosequence("ATCGTHCTCH")
    assert s.cut("T2", "T8") == "TCGTHCT"


def test_biosequence_from_ncbi():
    accession = "NP_005148.2"
    s = AminoAcidSequence.from_ncbi(accession)
    assert accession in s.name
    assert s.metadata["accession"] == accession


def test_biosequences_from_ncbis():
    ss = AminoAcidSequence.from_ncbi("NP_005148.2", "NP_001607.1")
    assert len(ss) == 2
