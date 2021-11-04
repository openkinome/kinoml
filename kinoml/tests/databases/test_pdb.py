"""
Test pdb functionalities of `kinoml.databases`
"""
from contextlib import contextmanager
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
    "pdb_id, success",
    [
        (
            "4YNE",  # PDB and CIF format available
            True,
        ),
        (
            "1BOS",  # only CIF format available
            True,
        ),
        (
            "XXXX",  # wrong code
            False,
        ),
    ],
)
def test_download_pdb_structure(pdb_id, success):
    """Try to download PDB structures."""
    from tempfile import TemporaryDirectory

    from kinoml.databases.pdb import download_pdb_structure

    with TemporaryDirectory() as temporary_directory:
        assert download_pdb_structure(pdb_id, temporary_directory) == success
