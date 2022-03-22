"""
Test uniprot functionalities of `kinoml.databases`
"""
import pytest


@pytest.mark.parametrize(
    "uniprot_id, valid_uniprot_id",
    [
        (
            "P00519",
            True,
        ),
        (
            "O95271",
            True,
        ),
        (
            "PXXXXX",
            False,
        ),
    ],
)
def test_download_fasta_file(uniprot_id, valid_uniprot_id):
    """Check if UniProt entries can be downloaded in fasta format."""
    from kinoml.databases.uniprot import download_fasta_file

    success = False
    fasta_path = download_fasta_file(uniprot_id)
    if fasta_path:
        success = True
        with open(fasta_path, "r") as fasta_file:
            first_character = fasta_file.read()[0]
            assert first_character == ">"

    assert success == valid_uniprot_id
