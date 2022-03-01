"""
Test klifs functionalities of `kinoml.databases`
"""
from contextlib import contextmanager
import pytest

from kinoml.databases.klifs import klifs_kinase_from_uniprot_id


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "uniprot_id, expectation, klifs_kinase_id",
    [("P00519", does_not_raise(), 392,), ("XXXXX", pytest.raises(ValueError), 392,),],
)
def test_klifs_kinase_from_uniprot_id(uniprot_id, expectation, klifs_kinase_id):
    """Compare klifs kinase ID for expected value."""
    with expectation:
        kinase = klifs_kinase_from_uniprot_id(uniprot_id)
        assert kinase["kinase.klifs_id"] == klifs_kinase_id
