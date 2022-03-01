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
            "4YNE",
            PosixPath,
        ),  # PDB and CIF format available
        (
            "1BOS",
            PosixPath,
        ),  # only CIF format available
        (
            "XXXX",
            bool,
        ),  # wrong code
    ],
)
def test_download_pdb_structure(pdb_id, return_type):
    """Try to download PDB structures."""
    from tempfile import TemporaryDirectory

    from kinoml.databases.pdb import download_pdb_structure

    with TemporaryDirectory() as temporary_directory:
        assert isinstance(download_pdb_structure(pdb_id, temporary_directory), return_type)
