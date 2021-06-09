"""
Test ligand featurizers of `kinoml.modeling`
"""
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

# TODO: rename solution to be more precise, e.g. n_heavy_atoms or full docstrings
# TODO: add files for tests to tests/data/proteins/ and tests/data/molecules, see https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
# TODO: think about including tests that are supposed to fail, e.g. raise an exception
# TODO: look at Parmed for good tests

@pytest.mark.parametrize(
    "smiles, solution",
    [
        (
            "C1=CC=NC=C1",
            6,
        ),
        (
            "CCNCC",
            5,
        ),
    ],
)
def test_read_smiles(smiles, solution):
    """Compare number of atoms of interpreted SMILES."""
    molecule = read_smiles(smiles)
    assert molecule.NumAtoms() == solution


@pytest.mark.parametrize(
    "package, resource, solutions",
    [
        (
            "kinoml.data.molecules",
            "chloroform.sdf",
            [4],
        ),
        (
            "kinoml.data.molecules",
            "chloroform.sdf",
            [4],
        ),
        (
            "kinoml.data.molecules",
            "chloroform_acetamide.sdf",
            [4, 4],
        ),
        (
            "kinoml.data.molecules",
            "chloroform_acetamide.pdb",
            [4, 4],
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [2475],
        ),
    ],
)
def test_read_molecules(package, resource, solutions):
    """Compare number of read molecules as well as atoms of each interpreted molecule."""
    with resources.path(package, resource) as path:
        molecules = read_molecules(str(path))
    assert len(molecules) == len(solutions)
    for molecule, solution in zip(molecules, solutions):
        assert molecule.NumAtoms() == solution


@pytest.mark.parametrize(
    "path, solution",
    [
        (
            "4f8o_phases.mtz",
            396011,
        ),
    ],
)
def test_read_electron_density(path, solution):
    """Compare number of grip points in the interpreted electron density."""
    electron_density = read_electron_density(path)
    assert electron_density.GetSize() == solution


@pytest.mark.parametrize(
    "molecules, suffix, solutions",
    [
        (
            [read_smiles("CCC")],
            ".sdf",
            [3]
        ),
        (
            [read_smiles("CCC")],
            ".pdb",
            [3]
        ),
        (
            [read_smiles("COCC"), read_smiles("cccccc")],
            ".sdf",
            [4, 6]
        ),
        (
            [read_smiles("CCC"), read_smiles("cccccc")],
            ".pdb",
            [3, 6]
        ),
    ],
)
def test_write_molecules(molecules, suffix, solutions):
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
        assert _count_molecules(temp_file.name) == len(solutions)
        for i, (molecule, solution) in enumerate(zip(molecules, solutions)):
            assert _count_atoms(temp_file.name, i) == solution


# TODO: Add a README to data directory explaining whats great about 4f8o
@pytest.mark.parametrize(
    "package, resource, chain_id, solution",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            2430
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "B",
            45
        ),
    ],
)
def test_select_chain(package, resource, chain_id, solution):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    selection = select_chain(molecule, chain_id)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "package, resource, alternate_location, solution",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "1",
            2441
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "2",
            2441
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "3",
            2441
        ),
    ],
)
def test_select_altloc(package, resource, alternate_location, solution):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    selection = select_altloc(molecule, alternate_location)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "package, resource, exceptions, remove_water, solution",
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
def test_remove_non_protein(package, resource, exceptions, remove_water, solution):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    selection = remove_non_protein(molecule, exceptions, remove_water)
    assert selection.NumAtoms() == solution


# TODO: removing a residue that is not there should raise an exception
@pytest.mark.parametrize(
    "package, resource, chain_id, residue_name, residue_id, solution",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            "GLY",
            22,
            2468
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "A",
            "ASP",
            22,
            2475
        ),
    ],
)
def test_delete_residue(package, resource, chain_id, residue_name, residue_id, solution):
    """Compare results to number of expected atoms."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    selection = delete_residue(molecule, chain_id, residue_name, residue_id)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "package, resource, solution",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            9
        ),
    ],
)
def test_get_expression_tags(package, resource, solution):
    """Compare results to number of expression tags."""
    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    expression_tags = get_expression_tags(molecule)
    assert len(expression_tags) == solution


@pytest.mark.parametrize(
    "package, resource, real_termini, solution",
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
            {"ACE"}
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            [138],
            {"ACE"}
        ),
    ],
)
def test_assign_caps(package, resource, real_termini, solution):
    """Compare results to number of expected atoms."""
    from openeye import oechem

    with resources.path(package, resource) as path:
        molecule = read_molecules(str(path))[0]
    molecule = assign_caps(molecule, real_termini)
    hier_view = oechem.OEHierView(molecule)
    caps = set([residue.GetResidueName() for residue in hier_view.GetResidues()
                if residue.GetResidueName() in ["ACE", "NME"]])
    assert caps == solution
