"""
Test OEDocking functionalities of `kinoml.docking`
"""
from contextlib import contextmanager
from importlib import resources
import pytest


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "package, resource, resids, expectation, min_x",
    [
        (
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            [50, 51, 52, 62, 63, 64, 70, 77],
            does_not_raise(),
            21.225000381469727,
        ),
        (
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            [700, 701, 702],
            pytest.raises(ValueError),
            21.225000381469727,
        ),
    ],
)
def test_resids_to_box_molecule(package, resource, resids, expectation, min_x):
    """Compare results to expected minimal x_coordinate."""
    from kinoml.modeling.OEModeling import read_molecules
    from kinoml.docking.OEDocking import resids_to_box_molecule

    with resources.path(package, resource) as path:
        with expectation:
            protein = read_molecules(str(path))[0]
            box_molecule = resids_to_box_molecule(protein, resids)
            x_coordinates = [coordinates[0] for coordinates in box_molecule.GetCoords().values()]
            assert round(min(x_coordinates), 3) == round(min_x, 3)
