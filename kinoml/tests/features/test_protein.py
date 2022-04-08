"""
Test ligand featurizers of `kinoml.protein`
"""
from importlib import resources


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
    featurizer = AminoAcidCompositionFeaturizer(use_multiprocessing=False)
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
    featurizer = OneHotEncodedSequenceFeaturizer(use_multiprocessing=False)
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
    featurizer = OneHotEncodedSequenceFeaturizer(
        sequence_type="klifs_kinase", use_multiprocessing=False
    )
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
    featurizer = OneHotEncodedSequenceFeaturizer(
        sequence_type="klifs_structure", use_multiprocessing=False
    )
    featurized_systems = featurizer.featurize(systems)

    assert len(featurized_systems) == 2  # needs structure_klifs_sequence or structure_klifs_id
    assert list(featurized_systems[0].featurizations["last"])[0][0] == 1
    assert list(featurized_systems[1].featurizations["last"])[0][14] == 1


def test_oeproteinstructurefeaturizer():
    """Check OEProteinStructureFeaturizer with different inputs."""
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OEProteinStructureFeaturizer

    systems = []
    # unspecifc definition of the system, only via PDB ID
    # modeling will be performed according to the sequence stored in the PDB Header
    protein = Protein(pdb_id="4f8o", name="PsaA")
    system = ProteinSystem(components=[protein])
    systems.append(system)
    # more specific definition of the system, protein of chain A co-crystallized with ligand AES
    # and alternate location B, modeling will be performed according to the sequence of the given
    # UniProt ID
    protein = Protein.from_pdb(pdb_id="4f8o", name="PsaA")
    protein.uniprot_id = "P31522"
    protein.chain_id = "A"
    protein.alternate_location = "B"
    protein.expo_id = "AES"
    system = ProteinSystem(components=[protein])
    systems.append(system)
    # use a protein structure form file
    with resources.path("kinoml.data.proteins", "4f8o_edit.pdb") as structure_path:
        protein = Protein.from_file(file_path=structure_path, name="PsaA")
        protein.uniprot_id = "P31522"
        system = ProteinSystem(components=[protein])
        systems.append(system)

    with resources.path("kinoml.data.proteins", "kinoml_tests_4f8o_spruce.loop_db") as loop_db:
        featurizer = OEProteinStructureFeaturizer(loop_db=loop_db)
        systems = featurizer.featurize(systems)
        # check number of residues
        assert len(systems[0].featurizations["last"].residues) == 239
        assert len(systems[1].featurizations["last"].residues) == 216
        assert len(systems[2].featurizations["last"].residues) == 109
        # check numbering of first residue
        assert systems[0].featurizations["last"].residues[0].resid == 1
        assert systems[1].featurizations["last"].residues[0].resid == 44
        assert systems[2].featurizations["last"].residues[0].resid == 47
