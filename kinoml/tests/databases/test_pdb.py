"""
Test pdb functionalities of `kinoml.databases`
"""
from contextlib import contextmanager
import pytest

from kinoml.databases.pdb import smiles_from_pdb


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
    with expectation:
        ligands = smiles_from_pdb(pdb_ids)
        for pdb_id, smiles in zip(pdb_ids, smiles_list):
            assert ligands[pdb_id] == smiles
