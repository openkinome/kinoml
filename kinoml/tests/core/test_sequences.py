"""
Test kinoml.core.sequences and derived objects
"""


def test_biosequence_mutation():
    from kinoml.core.sequences import Biosequence

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
    from kinoml.core.proteins import AminoAcidSequence

    s1 = AminoAcidSequence(uniprot_id="P00519-1")
    s2 = AminoAcidSequence(ncbi_id="NP_005148.2")
    assert s1.sequence == s2.sequence


def test_aminoacidsequence_fetching_with_alterations():
    from kinoml.core.proteins import AminoAcidSequence

    sequence = AminoAcidSequence(uniprot_id="P00519")
    assert len(sequence.sequence) == 1130
    assert sequence.sequence[314] == "T"

    sequence = AminoAcidSequence(
        uniprot_id="P00519",
        metadata={"construct_range": "229-512"}
    )
    assert len(sequence.sequence) == 284

    sequence = AminoAcidSequence(
        uniprot_id="P00519",
        metadata={"mutations": "T315A"}
    )
    assert sequence.sequence[314] == "A"

    sequence = AminoAcidSequence(
        uniprot_id="P00519",
        metadata={"mutations": "T315A del320-322P ins321AAA", "construct_range": "229-512"}
    )
    assert sequence.sequence[86] == "A"
    assert sequence.sequence[91] == "P"
    assert sequence.sequence[92:95] == "AAA"
    assert len(sequence.sequence) == 284
