"""
Test ligand featurizers of `kinoml.modeling`
"""
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
    "path, solutions",
    [
        (
            "chloroform.sdf",
            [4],
        ),
        (
            "chloroform.pdb",
            [4],
        ),
        (
            "chloroform_acetamide.sdf",
            [4, 4],
        ),
        (
            "chloroform_acetamide.pdb",
            [4, 4],
        ),
        (
            "4f8o.pdb",
            [2475],
        ),
    ],
)
def test_read_molecules(path, solutions):
    """Compare number of read molecules as well as atoms of each interpreted molecule."""
    molecules = read_molecules(path)
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
    "molecule, chain_id, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            "A",
            2430
        ),
        (
            read_molecules("4f8o.pdb")[0],
            "B",
            45
        ),
    ],
)
def test_select_chain(molecule, chain_id, solution):
    """Compare results to number of expected atoms."""
    selection = select_chain(molecule, chain_id)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "molecule, alternate_location, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            1,
            2441
        ),
        (
            read_molecules("4f8o.pdb")[0],
            2,
            2441
        ),
        (
            read_molecules("4f8o.pdb")[0],
            3,
            2441
        ),
    ],
)
def test_select_altloc(molecule, alternate_location, solution):
    """Compare results to number of expected atoms."""
    selection = select_altloc(molecule, alternate_location)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "molecule, exceptions, remove_water, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            [],
            True,
            2104
        ),
        (
            read_molecules("4f8o.pdb")[0],
            [],
            False,
            2393
        ),
        (
            read_molecules("4f8o.pdb")[0],
            ["AES"],
            True,
            2122
        ),
    ],
)
def test_remove_non_protein(molecule, exceptions, remove_water, solution):
    """Compare results to number of expected atoms."""
    selection = remove_non_protein(molecule, exceptions, remove_water)
    assert selection.NumAtoms() == solution


# TODO: removing a residue that is not there should raise an exception
@pytest.mark.parametrize(
    "molecule, chain_id, residue_name, residue_id, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            "A",
            "GLY",
            22,
            2468
        ),
        (
            read_molecules("4f8o.pdb")[0],
            "A",
            "ASP",
            22,
            2475
        ),
    ],
)
def test_delete_residue(molecule, chain_id, residue_name, residue_id, solution):
    """Compare results to number of expected atoms."""
    selection = delete_residue(molecule, chain_id, residue_name, residue_id)
    assert selection.NumAtoms() == solution


@pytest.mark.parametrize(
    "molecule, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            9
        ),
    ],
)
def test_get_expression_tags(molecule, solution):
    """Compare results to number of expression tags."""
    expression_tags = get_expression_tags(molecule)
    assert len(expression_tags) == solution


@pytest.mark.parametrize(
    "molecule, real_termini, solution",
    [
        (
            read_molecules("4f8o.pdb")[0],
            [],
            {"ACE", "NME"}
        ),
        (
            read_molecules("4f8o.pdb")[0],
            [1, 138],
            set()
        ),
        (
            read_molecules("4f8o.pdb")[0],
            [1],
            {"ACE"}
        ),
        (
            read_molecules("4f8o.pdb")[0],
            [138],
            {"ACE"}
        ),
    ],
)
def test_assign_caps(molecule, real_termini, solution):
    """Compare results to number of expected atoms."""
    from openeye import oechem

    molecule = assign_caps(molecule, real_termini)
    hier_view = oechem.OEHierView(molecule)
    caps = set([residue.GetResidueName() for residue in hier_view.GetResidues()
                if residue.GetResidueName() in ["ACE", "NME"]])
    assert caps == solution
