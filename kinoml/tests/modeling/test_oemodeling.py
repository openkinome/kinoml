"""
Test OEModeling functionalities of `kinoml.modeling`
"""
from contextlib import contextmanager
from importlib import resources
import pytest
import tempfile

from kinoml.modeling.OEModeling import (
    read_smiles,
    read_molecules,
    read_electron_density,
    write_molecules,
    select_chain,
    select_altloc,
    remove_non_protein,
    delete_residue,
    get_expression_tags,
    assign_caps
)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "smiles, add_hydrogens, expectation, n_atoms",
    [
        (
            "C1=CC=NC=C1",
            True,
            does_not_raise(),
            11,
        ),
        (
            "C1=CC=[NH+]C=C1",
            True,
            does_not_raise(),
            12,
        ),
        (
            "CCNCC",
            True,
            does_not_raise(),
            16,
        ),
        (
            "CCNCC",
            False,
            does_not_raise(),
            5,
        ),
        (
            "1",
            False,
            pytest.raises(ValueError),
            0
        ),
    ],
)
def test_read_smiles(smiles, add_hydrogens, expectation, n_atoms):
    """Compare number of atoms of interpreted SMILES."""
    with expectation:
        molecule = read_smiles(smiles, add_hydrogens)
        assert molecule.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "package, resource, add_hydrogens, expectation, n_atoms_list",
    [
        (
            "kinoml.data.molecules",
            "chloroform.sdf",
            False,
            does_not_raise(),
            [4],
        ),
        (
            "kinoml.data.molecules",
            "chloroform.sdf",
            True,
            does_not_raise(),
            [5],
        ),
        (
            "kinoml.data.molecules",
            "chloroform_acetamide.sdf",
            True,
            does_not_raise(),
            [5, 9],
        ),
        (
            "kinoml.data.molecules",
            "chloroform_acetamide.pdb",
            True,
            does_not_raise(),
            [5, 9],
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            True,
            does_not_raise(),
            [2497],
        ),
        (
            "kinoml.data.electron_densities",
            "4f8o_phases.mtz",
            True,
            pytest.raises(ValueError),
            [],
        ),
    ],
)
def test_read_molecules(package, resource, add_hydrogens, expectation, n_atoms_list):
    """Compare number of read molecules as well as atoms of each interpreted molecule."""
    with resources.path(package, resource) as path:
        with expectation:
            molecules = read_molecules(str(path), add_hydrogens)
            assert len(molecules) == len(n_atoms_list)
            for molecule, n_atmos in zip(molecules, n_atoms_list):
                assert molecule.NumAtoms() == n_atmos


@pytest.mark.parametrize(
    "package, resource, expectation, n_grid_points",
    [
        (
            "kinoml.data.electron_densities",
            "4f8o_phases.mtz",
            does_not_raise(),
            396011,
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            pytest.raises(ValueError),
            0,
        ),
    ],
)
def test_read_electron_density(package, resource, expectation, n_grid_points):
    """Compare number of grip points in the interpreted electron density."""
    with resources.path(package, resource) as path:
        with expectation:
            electron_density = read_electron_density(path)
            assert electron_density.GetSize() == n_grid_points


@pytest.mark.parametrize(
    "molecules, suffix, n_atoms_list",
    [
        (
            [read_smiles("CCC")],
            ".sdf",
            [11]
        ),
        (
            [read_smiles("CCC")],
            ".pdb",
            [11]
        ),
        (
            [read_smiles("COCC"), read_smiles("cccccc")],
            ".sdf",
            [12, 14]
        ),
        (
            [read_smiles("CCC"), read_smiles("cccccc")],
            ".pdb",
            [11, 14]
        ),
    ],
)
def test_write_molecules(molecules, suffix, n_atoms_list):
    """Compare number of molecules and atoms in the written file."""
    def _count_molecules(path):
        with open(path) as rf:
            if path.split(".")[-1] == "sdf":
                return rf.read().count("\n$$$$\n")
            elif path.split(".")[-1] == "pdb":
                return rf.read().count("\nEND\n")
            else:
                raise NotImplementedError

    def _count_atoms(path, index):
        with open(path) as rf:
            if path.split(".")[-1] == "sdf":
                molecule_text = rf.read().split("\n$$$$\n")[index]
                return int(molecule_text.split("\n")[3].split()[0])
            elif path.split(".")[-1] == "pdb":
                molecule_text = rf.read().split("\nEND\n")[index]
                return len([line for line in molecule_text.split("\n")
                            if line.startswith(("ATOM", "HETATM"))])
            else:
                raise NotImplementedError

    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        write_molecules(molecules, temp_file.name)
        assert _count_molecules(temp_file.name) == len(n_atoms_list)
        for i, (molecule, n_atoms) in enumerate(zip(molecules, n_atoms_list)):
            assert _count_atoms(temp_file.name, i) == n_atoms


# TODO: Add a README to data directory explaining whats great about 4f8o
@pytest.mark.parametrize(
    "package, resource, chain_id, expectation, n_atoms",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            does_not_raise(),
            2430
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "B",
            does_not_raise(),
            45
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "1",
            pytest.raises(ValueError),
            0
        ),
    ],
)
def test_select_chain(package, resource, chain_id, expectation, n_atoms):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
        with expectation:
            selection = select_chain(molecule, chain_id)
            assert selection.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "package, resource, alternate_location, expectation, n_atoms",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            does_not_raise(),
            2458
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "B",
            does_not_raise(),
            2458
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "C",
            pytest.raises(ValueError),
            2458
        ),
    ],
)
def test_select_altloc(package, resource, alternate_location, expectation, n_atoms):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
        with expectation:
            selection = select_altloc(molecule, alternate_location)
            assert selection.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "package, resource, exceptions, remove_water, n_atoms",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [],
            True,
            2104
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [],
            False,
            2393
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            ["AES"],
            True,
            2122
        ),
    ],
)
def test_remove_non_protein(package, resource, exceptions, remove_water, n_atoms):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    selection = remove_non_protein(molecule, exceptions, remove_water)
    assert selection.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "package, resource, chain_id, residue_name, residue_id, expectation, n_atoms",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            "GLY",
            22,
            does_not_raise(),
            2468
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            "ASP",
            22,
            pytest.raises(ValueError),
            2468
        ),
    ],
)
def test_delete_residue(package, resource, chain_id, residue_name, residue_id, expectation, n_atoms):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        with expectation:
            molecule = read_molecules(str(path))[0]
            selection = delete_residue(molecule, chain_id, residue_name, residue_id)
            assert selection.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "package, resource, n_expression_tags",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            9
        ),
    ],
)
def test_get_expression_tags(package, resource, n_expression_tags):
    """Compare results to number of expression tags."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    expression_tags = get_expression_tags(molecule)
    assert len(expression_tags) == n_expression_tags


@pytest.mark.parametrize(
    "package, resource, real_termini, caps",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [],
            {"ACE", "NME"}
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [1, 138],
            set()
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [1],
            {"NME"}
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [138],
            {"ACE"}
        ),
    ],
)
def test_assign_caps(package, resource, real_termini, caps):
    """Compare results to expected caps."""
    from openeye import oechem

    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
        molecule = select_altloc(molecule, 'A')
        molecule = assign_caps(molecule, real_termini)
        hier_view = oechem.OEHierView(molecule)
        found_caps = set(
            [
                residue.GetResidueName() for residue in hier_view.GetResidues()
                if residue.GetResidueName() in ["ACE", "NME"]
            ]
        )
        assert found_caps == caps
