"""
Test ligand featurizers of `kinoml.protein`
"""
from importlib import resources
import pytest


@pytest.mark.parametrize(
    "package, resource_list, pdb_id, uniprot_id, chain_id, alternate_location, expo_id, "
    "expected_n_residues",
    [
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db"],
            "4f8o",
            "P31522",
            "A",
            "B",
            "AES",
            216
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.pdb"],
            None,
            None,
            None,
            None,
            None,
            239
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.cif"],
            None,
            None,
            None,
            None,
            None,
            240  # TODO: adds cap on residue 1, submit bug report
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o_edit.pdb"],
            None,
            "P31522",
            None,
            None,
            None,
            109
        ),
    ],
)
def test_OEProteinStructureFeaturizer(
        package,
        resource_list,
        pdb_id,
        uniprot_id,
        chain_id,
        alternate_location,
        expo_id,
        expected_n_residues
):
    """
    Compare featurizer results to expected number of residues.
    """
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OEProteinStructureFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEProteinStructureFeaturizer(loop_db=loop_db)
        base_protein = BaseProtein(name="PsaA")
        if uniprot_id:
            base_protein.uniprot_id = uniprot_id
        if chain_id:
            base_protein.chain_id = chain_id
        if alternate_location:
            base_protein.alternate_location = alternate_location
        if expo_id:
            base_protein.expo_id = expo_id
        if pdb_id:
            base_protein.pdb_id = pdb_id
            system = ProteinSystem([base_protein])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEProteinStructureFeaturizer"].universe
            n_residues = u.residues.n_residues
            assert n_residues == expected_n_residues
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinSystem([base_protein])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEProteinStructureFeaturizer"].universe
                n_residues = u.residues.n_residues
                assert n_residues == expected_n_residues


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
