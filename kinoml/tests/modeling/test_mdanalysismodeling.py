"""
Test OEModeling functionalities of `kinoml.modeling`
"""
from contextlib import contextmanager
from importlib import resources
import pytest


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "package, resource, expectation, n_atoms",
    [
        (  # unsupported file format
            "kinoml.data.molecules",
            "chloroform_acetamide.sdf",
            pytest.raises(ValueError),
            14,
        ),
        (  # multi-molecule pdb
            "kinoml.data.molecules",
            "chloroform_acetamide.pdb",
            pytest.raises(IndexError),
            14,
        ),
        (  # correct pdb
            "kinoml.data.proteins",
            "4f8o.pdb",
            does_not_raise(),
            2475,
        ),
    ],
)
def test_read_molecule(package, resource, expectation, n_atoms):
    """Compare results to expected number of atoms."""
    from kinoml.modeling.MDAnalysisModeling import read_molecule

    with resources.path(package, resource) as path:
        with expectation:
            molecule = read_molecule(str(path))
            assert len(molecule.atoms) == n_atoms
