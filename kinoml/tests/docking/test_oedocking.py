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


@pytest.mark.parametrize(
    "package, resource, smiles_list, n_poses",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            ["c1cc(ccc1CCN)S(=O)(=O)F", "c1cc(ccc1CCN)S(=O)(=O)N"],
            3,
        ),
    ],
)
def test_hybrid_docking(package, resource, smiles_list, n_poses):
    """Compare results to expected number of docked molecules and docking poses"""
    from openeye import oedocking

    from kinoml.docking.OEDocking import hybrid_docking
    from kinoml.modeling.OEModeling import read_molecules, read_smiles, prepare_complex

    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        design_unit = prepare_complex(structure)
        if not design_unit.HasReceptor():
            oedocking.OEMakeReceptor(design_unit)
        docking_poses = hybrid_docking(
            design_unit, [read_smiles(smiles) for smiles in smiles_list], n_poses
        )
        assert len(docking_poses) == len(smiles_list) * n_poses


@pytest.mark.parametrize(
    "package, resource, resids, smiles_list, n_poses",
    [
        (
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            [50, 51, 52, 62, 63, 64, 70, 77],
            ["c1cc(ccc1CCN)S(=O)(=O)F", "c1cc(ccc1CCN)S(=O)(=O)N"],
            3,
        ),
    ],
)
def test_chemgauss_docking(package, resource, resids, smiles_list, n_poses):
    """Compare results to expected number of docked molecules and docking poses"""
    from openeye import oechem, oedocking

    from kinoml.docking.OEDocking import chemgauss_docking, resids_to_box_molecule
    from kinoml.modeling.OEModeling import read_molecules, read_smiles, prepare_protein

    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        design_unit = prepare_protein(structure)
        protein = oechem.OEGraphMol()
        design_unit.GetProtein(protein)
        box_molecule = resids_to_box_molecule(protein, resids)
        options = oedocking.OEMakeReceptorOptions()
        options.SetBoxMol(box_molecule)
        oedocking.OEMakeReceptor(design_unit, options)
        docking_poses = chemgauss_docking(
            design_unit, [read_smiles(smiles) for smiles in smiles_list], n_poses
        )
        assert len(docking_poses) == len(smiles_list) * n_poses


@pytest.mark.parametrize(
    "package, resource, smiles_list",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            ["c1cc(ccc1CCN)S(=O)(=O)F", "c1cc(ccc1CCN)S(=O)(=O)N"],
        ),
    ],
)
def test_pose_molecules(package, resource, smiles_list):
    """Compare results to expected number of docked molecules and docking poses"""
    from openeye import oedocking

    from kinoml.docking.OEDocking import pose_molecules
    from kinoml.modeling.OEModeling import read_molecules, read_smiles, prepare_complex

    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        design_unit = prepare_complex(structure)
        if not design_unit.HasReceptor():
            oedocking.OEMakeReceptor(design_unit)
        docking_poses = pose_molecules(
            design_unit, [read_smiles(smiles) for smiles in smiles_list]
        )
        assert len(docking_poses) == len(smiles_list)
