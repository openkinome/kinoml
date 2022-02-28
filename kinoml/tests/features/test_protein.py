"""
Test protein featurizers of `kinoml.features`
"""


def test_aminoacidcompositionfeaturizer():
    """Check AminoAcidCompositionFeaturizer."""
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import AminoAcidCompositionFeaturizer

    systems = [
        ProteinSystem([Protein(sequence="")]),
        ProteinSystem([Protein(sequence="A")]),
        ProteinSystem([Protein(uniprot_id="P00519")]),
        ProteinSystem([Protein(uniprot_id="xxxxx")]),
    ]
    featurizer = AminoAcidCompositionFeaturizer()
    featurized_systems = featurizer.featurize(systems)

    assert len(featurized_systems) == 3  # filter protein with wrong UniProt ID
    assert list(featurized_systems[0].featurizations["last"]) == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert list(featurized_systems[1].featurizations["last"]) == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert list(featurized_systems[2].featurizations["last"]) == [
        97,
        14,
        42,
        90,
        29,
        88,
        24,
        33,
        82,
        96,
        18,
        40,
        90,
        33,
        64,
        120,
        65,
        62,
        13,
        30,
    ]


def test_onehotencodedsequencefeaturizer_full():
    """Check OneHotEncodedSequenceFeaturizer with full sequence."""
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OneHotEncodedSequenceFeaturizer

    systems = [
        ProteinSystem([Protein(sequence="")]),
        ProteinSystem([Protein(sequence="A")]),
        ProteinSystem([Protein(uniprot_id="P00519")]),
        ProteinSystem([Protein(uniprot_id="xxxxx")]),
    ]
    featurizer = OneHotEncodedSequenceFeaturizer()
    featurized_systems = featurizer.featurize(systems)

    assert len(featurized_systems) == 2  # filter protein with wrong UniProt ID and empty string
    assert list(featurized_systems[0].featurizations["last"])[0][0] == 1
    assert list(featurized_systems[1].featurizations["last"])[3][2] == 1


def test_onehotencodedsequencefeaturizer_klifs_kinase():
    """Check OneHotEncodedSequenceFeaturizer with kinase KLIFS sequence."""
    from kinoml.core.proteins import KLIFSKinase
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OneHotEncodedSequenceFeaturizer

    systems = [
        ProteinSystem([KLIFSKinase(sequence="")]),
        ProteinSystem([KLIFSKinase(kinase_klifs_sequence="A")]),
        ProteinSystem([KLIFSKinase(uniprot_id="P00519")]),
        ProteinSystem([KLIFSKinase(uniprot_id="xxxxx")]),
        ProteinSystem([KLIFSKinase(ncbi_id="NP_005148.2")]),
        ProteinSystem([KLIFSKinase(kinase_klifs_id=480)]),
        ProteinSystem([KLIFSKinase(structure_klifs_id=3620)]),
    ]
    featurizer = OneHotEncodedSequenceFeaturizer(sequence_type="klifs_kinase")
    featurized_systems = featurizer.featurize(systems)

    assert len(featurized_systems) == 5  # filter protein with wrong UniProt ID and empty string
    assert list(featurized_systems[0].featurizations["last"])[0][0] == 1
    assert list(featurized_systems[1].featurizations["last"])[0][14] == 1
    assert list(featurized_systems[2].featurizations["last"])[0][14] == 1
    assert list(featurized_systems[3].featurizations["last"])[0][14] == 1
    assert list(featurized_systems[4].featurizations["last"])[0][14] == 1


def test_onehotencodedsequencefeaturizer_klifs_structure():
    """Check OneHotEncodedSequenceFeaturizer with structure KLIFS sequence."""
    from kinoml.core.proteins import KLIFSKinase
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OneHotEncodedSequenceFeaturizer

    systems = [
        ProteinSystem([KLIFSKinase(sequence="")]),
        ProteinSystem([KLIFSKinase(structure_klifs_sequence="A")]),
        ProteinSystem([KLIFSKinase(uniprot_id="P00519")]),
        ProteinSystem([KLIFSKinase(kinase_klifs_id=480)]),
        ProteinSystem([KLIFSKinase(structure_klifs_id=3620)]),
    ]
    featurizer = OneHotEncodedSequenceFeaturizer(sequence_type="klifs_structure")
    featurized_systems = featurizer.featurize(systems)

    assert len(featurized_systems) == 2  # needs structure_klifs_sequence or structure_klifs_id
    assert list(featurized_systems[0].featurizations["last"])[0][0] == 1
    assert list(featurized_systems[1].featurizations["last"])[0][14] == 1
