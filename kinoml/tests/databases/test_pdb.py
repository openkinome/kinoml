"""
Test pdb functionalities of `kinoml.databases`
"""
from contextlib import contextmanager
from pathlib import PosixPath
import pytest


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "pdb_ids, expectation, smiles_list",
    [
        (
            ["EDO"],
            does_not_raise(),
            ["C(CO)O"],
        ),
        (
            ["---"],
            pytest.raises(KeyError),
            ["---"],
        ),
        (
            ["EDO", "GOL"],
            does_not_raise(),
            ["C(CO)O", "C(C(CO)O)O"],
        ),
    ],
)
def test_smiles_from_pdb(pdb_ids, expectation, smiles_list):
    """Compare results for expected SMILES."""
    from kinoml.databases.pdb import smiles_from_pdb

    with expectation:
        ligands = smiles_from_pdb(pdb_ids)
        for pdb_id, smiles in zip(pdb_ids, smiles_list):
            assert ligands[pdb_id] == smiles


@pytest.mark.parametrize(
    "pdb_id, return_type",
    [
        (
            "4YNE",  # PDB and CIF format available
            PosixPath,
        ),
        (
            "1BOS",  # only CIF format available
            PosixPath,
        ),
        (
            "XXXX",  # wrong code
            bool,
        ),
    ],
)
def test_download_pdb_structure(pdb_id, return_type):
    """Try to download PDB structures."""
    from tempfile import TemporaryDirectory

    from kinoml.databases.pdb import download_pdb_structure

    with TemporaryDirectory() as temporary_directory:
        assert isinstance(download_pdb_structure(pdb_id, temporary_directory), return_type)


@pytest.mark.parametrize(
    "pdb_id, chain_id, expo_id, smiles, return_type",
    [
        (
            "4YNE",  # PDB and CIF format available
            "A",
            "4EK",
            "c1ccnc(c1)c2cnc3n2nc(cc3)N4CCC[C@@H]4c5cccc(c5)F",
            PosixPath,
        ),
        (
            "1BOS",  # only CIF format available
            "E",
            "GAL",
            "C([C@@H]1[C@@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)O",
            PosixPath,
        ),
        (
            "XXXX",  # wrong code
            "X",
            "XXX",
            "xxxxx",
            bool,
        ),
    ],
)
def test_download_pdb_structure(pdb_id, chain_id, expo_id, smiles, return_type):
    """Try to download PDB ligands."""
    from tempfile import TemporaryDirectory
    from kinoml.databases.pdb import download_pdb_ligand

    with TemporaryDirectory() as temporary_directory:
        assert isinstance(download_pdb_ligand(pdb_id, chain_id, expo_id, smiles), return_type)
