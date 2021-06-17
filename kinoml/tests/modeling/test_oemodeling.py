"""
Test OEModeling functionalities of `kinoml.modeling`
"""
from contextlib import contextmanager
from importlib import resources
import pytest
import tempfile

from bravado_core.exception import SwaggerMappingError

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
    assign_caps,
    prepare_structure,
    read_klifs_ligand,
    generate_tautomers,
    generate_enantiomers,
    generate_conformations,
    generate_reasonable_conformations,
    overlay_molecules,
    enumerate_isomeric_smiles,
    are_identical_molecules,
    get_sequence,
    get_structure_sequence_alignment,
    apply_deletions,
    apply_insertions,
    apply_mutations,
    delete_partial_residues,
    delete_short_protein_segments,
    delete_clashing_sidechains,
    get_atom_coordinates,
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
    """Compare results to expected number of atoms."""
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
    """Compare results to expected number of read molecules as well as atoms of each interpreted molecule."""
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
    """Compare results to expected number of grip points in the interpreted electron density."""
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
    """Compare results to expected number of molecules and atoms in the written file."""
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
    """Compare results to expected number of atoms."""
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
    """Compare results to expected number of atoms."""
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
    """Compare results to expected number of atoms."""
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
    """Compare results to expected number of expression tags."""
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

@pytest.mark.parametrize(
    "package, resource, has_ligand, chain_id, altloc, ligand_name, expectation, title_contains",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            True,
            "A",
            "A",
            "AES",
            does_not_raise(),
            ["(A)", "AES"]
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            False,
            "A",
            "A",
            None,
            does_not_raise(),
            ["(A)"]
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            True,
            "A",
            "C",
            "AES",
            pytest.raises(ValueError),
            ["(A)", "AES"]
        ),
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            True,
            "C",
            "A",
            "AES",
            pytest.raises(ValueError),
            ["(C)", "AES"]
        ),
    ],
)
def test_prepare_structure(package, resource, has_ligand, chain_id, altloc, ligand_name, expectation, title_contains):
    """Check if returned design unit title contains expected strings."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        with expectation:
            design_unit = prepare_structure(
                structure,
                has_ligand=has_ligand,
                chain_id=chain_id,
                alternate_location=altloc,
                ligand_name=ligand_name
            )
            assert all(x in design_unit.GetTitle() for x in title_contains)


@pytest.mark.parametrize(
    "klifs_structure_id, expectation, n_atoms",
    [
        (
            1104,
            does_not_raise(),
            45
        ),
        (
            1045,
            pytest.raises(ValueError),
            0
        ),
        (
            "X",
            pytest.raises(SwaggerMappingError),
            0
        ),
    ],
)
def test_read_klifs_ligand(klifs_structure_id, expectation, n_atoms):
    """Compare results to expected number of atoms."""
    with expectation:
        molecule = read_klifs_ligand(klifs_structure_id)
        assert molecule.NumAtoms() == n_atoms


@pytest.mark.parametrize(
    "smiles, n_tautomers",
    [
        (
            "COC",
            1
        ),
        (
            "CCC(O)C(C)=O",
            2
        ),
        (
            r"C\N=C\NCC(O)C(C)=O",
            4
        ),
        (
            r"C\N=C/NCCC(=O)C(O)CC(CN\C=N\C)C(O)C(=O)CCN\C=N\C",
            16
        ),
    ],
)
def test_generate_tautomers(smiles, n_tautomers):
    """Compare results to expected number of tautomers."""
    molecule = read_smiles(smiles)
    tautomers = generate_tautomers(molecule)
    assert len(tautomers) == n_tautomers


@pytest.mark.parametrize(
    "smiles, n_enantiomers",
    [
        (
            "CC(C)(C)C",
            1
        ),
        (
            "C(C)(F)Cl",
            2
        ),
        (
            "CC(Cl)CCC(O)F",
            4
        ),
        (
            "CC(Cl)CC(C)C(O)F",
            8
        ),
    ],
)
def test_generate_enantiomers(smiles, n_enantiomers):
    """Compare results to expected number of enantiomers."""
    molecule = read_smiles(smiles)
    enantiomers = generate_enantiomers(molecule)
    assert len(enantiomers) == n_enantiomers


@pytest.mark.parametrize(
    "smiles, n_conformations",
    [
        (
            "CCC(C)C(C)=O",
            5
        ),
        (
            "C1CCN(C1)CCOC2=C3COCC=CCOCC4=CC(=CC=C4)C5=NC(=NC=C5)NC(=C3)C=C2",
            5
        ),
    ],
)
def test_generate_conformations(smiles, n_conformations):
    """Compare results to expected number of conformations."""
    molecule = read_smiles(smiles)
    conformations = generate_conformations(molecule, max_conformations=5)
    assert conformations.NumConfs() == n_conformations


@pytest.mark.parametrize(
    "smiles, n_conformations_list",
    [
        (
            "FC(Cl)Br",
            [1, 1]
        ),
        (
            "CC(Cl)CCC(O)F",
            [5, 5, 5, 5]
        ),
    ],
)
def test_generate_reasonable_conformations(smiles, n_conformations_list):
    """Compare results to expected number of isomers and conformations."""
    molecule = read_smiles(smiles)
    conformations_ensemble = generate_reasonable_conformations(molecule, max_conformations=5)
    assert len(conformations_ensemble) == len(n_conformations_list)
    for conformations, n_conformations in zip(conformations_ensemble, n_conformations_list):
        assert conformations.NumConfs() == n_conformations


@pytest.mark.parametrize(
    "reference_smiles, fit_smiles, comparator",
    [
        (
            "C1=CC=C(C=C1)C1=CC=CC=C1",
            "S1C=NC=C1C1=CC=CC=C1",
            ">"
        ),
        (
            "C1=CC=CC=C1",
            "COC",
            "<"
        ),
    ],
)
def test_overlay_molecules(reference_smiles, fit_smiles, comparator):
    """Compare results to have a TanimotoCombo score bigger or smaller than 1."""
    reference_molecule = read_smiles(reference_smiles)
    reference_molecule = generate_conformations(reference_molecule, max_conformations=1)
    fit_molecule = read_smiles(fit_smiles)
    fit_molecule = generate_conformations(fit_molecule, max_conformations=10)
    score, overlay = overlay_molecules(reference_molecule, fit_molecule)
    if comparator == ">":
        assert score > 1
    elif comparator == "<":
        assert score < 1
    else:
        raise ValueError("Wrong comparator provided. Only '<' and '>' are allowed.")


@pytest.mark.parametrize(
    "smiles, n_smiles",
    [
        (
            "CC(=O)C(C)O",
            2
        ),
        (
            "CCC(=O)C(C)O",
            4
        ),
        (
            "C[C@H](F)Cl",
            1
        ),
        (
            "CC(F)Cl",
            2
        ),
    ],
)
def test_enumerate_isomeric_smiles(smiles, n_smiles):
    """Compare results to expected number of generated isomeric SMILES strings."""
    molecule = read_smiles(smiles)
    isomeric_smiles_representations = enumerate_isomeric_smiles(molecule)
    assert len(isomeric_smiles_representations) == n_smiles


@pytest.mark.parametrize(
    "smiles1, smiles2, identical_molecules",
    [
        (
            "CC(=O)C(C)O",
            "C[C@@H](O)C(C)=O",
            True
        ),
        (
            "CCC(=O)C(C)O",
            "CC[C@@H](O)C(C)=O",
            True
        ),
        (
            "C[C@H](F)Cl",
            "CC(F)Cl",
            True
        ),
        (
            "C[C@H](F)Cl",
            "C[C@@H](F)Cl",
            False
        ),
    ],
)
def test_are_identical_molecules(smiles1, smiles2, identical_molecules):
    """Compare results to expected molecular identity."""
    molecule1 = read_smiles(smiles1)
    molecule2 = read_smiles(smiles2)
    assert are_identical_molecules(molecule1, molecule2) == identical_molecules


@pytest.mark.parametrize(
    "package, resource, sequence",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGLXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        ),
        (
            "kinoml.data.molecules",
            "chloroform.pdb",
            "X",
        ),
    ],
)
def test_get_sequence(package, resource, sequence):
    """Compare results to expected sequence."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        assert get_sequence(structure) == sequence


@pytest.mark.parametrize(
    "package, resource, sequence, expected_alignment",
    [
        (  # mutation (middle and end)
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGV",
            [
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGV"
            ]
        ),
        (  # insertions (missing D82 could be placed at two positions, only "D-" is correct)
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGVVVGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGLEHHHHHH",
            [
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKG---GYMISADGDYVGLYSYMMSWVGIDNNWYIND-SPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTV-QGL-------",
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGVVVGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGLEHHHHHH"
            ]
        ),
        (  # deletions (start and middle)
            "kinoml.data.proteins",
            "4f8o.pdb",
            "FHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            [
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
                "---FHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGD---LYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL"
            ]
        ),
        (  # all together
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            "FHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGVVVGYMISADGDLFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGVEHHHHHH",
            [
                "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKG---GYMISADGDYVGLYSYMMSWVGIDNNWYIND-SPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTV-QGL-------",
                "---FHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGVVVGYMISADGD---LFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGVEHHHHHH"
            ]
        )
    ],
)
def test_get_structure_sequence_alignment(package, resource, sequence, expected_alignment):
    """Compare results to expected sequence alignment."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        structure = remove_non_protein(structure, remove_water=True)
        alignment = get_structure_sequence_alignment(structure, sequence)
        for sequence1, sequence2 in zip(alignment, expected_alignment):
            assert sequence1 == sequence2


@pytest.mark.parametrize(
    "package, resource, sequence, delete_n_anchors, expectation, expected_sequence",
    [
        (  # no deletions, ignore mutations
            "kinoml.data.proteins",
            "4f8o.pdb",
            "XNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGXXXISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQXX",
            2,
            does_not_raise(),
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
        (  # delete residue 4 including two anchoring residues on both sides
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            2,
            does_not_raise(),
            "MDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
        (  # delete first residue, anchoring residues will be ignored, since it's at the beginning
            "kinoml.data.proteins",
            "4f8o.pdb",
            "NTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            2,
            does_not_raise(),
            "NTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
        (  # delete residue 4 without anchoring residues
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            0,
            does_not_raise(),
            "MNTHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
        (  # negative value for 'delete_n_anchors' should raise ValueError
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            -1,
            pytest.raises(ValueError),
            "MNTHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        )
    ],
)
def test_apply_deletions(package, resource, sequence, delete_n_anchors, expectation, expected_sequence):
    """Compare results to expected sequence."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        structure = remove_non_protein(structure, remove_water=True)
        with expectation:
            structure_with_deletions = apply_deletions(structure, sequence, delete_n_anchors)
            sequence_with_deletions = get_sequence(structure_with_deletions)
            assert sequence_with_deletions == expected_sequence


@pytest.mark.parametrize(
    "package_list, resource_list, sequence",
    [
        (  # test loop modeling for deleted residue D82
            ["kinoml.data.proteins", "kinoml.data.proteins"],
            ["4f8o_edit.pdb", "kinoml_tests_4f8o_spruce.loop_db"],
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVQGL",
        )
    ],
)
def test_apply_insertions(package_list, resource_list, sequence):
    """Compare results to expected sequence."""
    with resources.path(package_list[0], resource_list[0]) as pdb_path:
        with resources.path(package_list[1], resource_list[1]) as loop_db_path:
            structure = read_molecules(str(pdb_path))[0]
            structure = remove_non_protein(structure, remove_water=True)
            structure_with_insertions = apply_insertions(structure, sequence, loop_db_path)
            sequence_with_insertions = get_sequence(structure_with_insertions)
            assert sequence_with_insertions == sequence


@pytest.mark.parametrize(
    "package, resource, sequence, delete_fallback, expectation, expected_sequence",
    [
        (  # mutation succeeds (middle GLYSY -> GLFSY)
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            True,
            does_not_raise(),
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLFSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL"
        ),
        (  # mutation fails with delete_fallback (middle TMGDT -> TMWDT)
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMWDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            True,
            does_not_raise(),
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
        (  # mutation fails without delete_fallback (middle TMGDT -> TMWDT)
            "kinoml.data.proteins",
            "4f8o.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMWDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
            False,
            pytest.raises(ValueError),
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMWDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVKQGL",
        ),
    ],
)
def test_apply_mutations(package, resource, sequence, delete_fallback, expectation, expected_sequence):
    """Compare results to expected sequence."""
    with resources.path(package, resource) as pdb_path:
        structure = read_molecules(str(pdb_path))[0]
        structure = remove_non_protein(structure, remove_water=True)
        with expectation:
            structure_with_mutations = apply_mutations(structure, sequence, delete_fallback)
            sequence_with_mutations = get_sequence(structure_with_mutations)
            assert sequence_with_mutations == expected_sequence


@pytest.mark.parametrize(
    "package, resource, delete_backbone_C, sequence",
    [
        (  # delete first two (missing sidechains) and last (missing backbone C) residue
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            "LEU138",
            "TFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVQG",
        ),
    ],
)
def test_delete_partial_residues(package, resource, delete_backbone_C, sequence):
    """Compare results to expected sequence."""
    from openeye import oechem

    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        if delete_backbone_C:
            hier_view = oechem.OEHierView(structure)
            hier_residue = hier_view.GetResidue("A", delete_backbone_C[:3], int(delete_backbone_C[3:]))
            for atom in hier_residue.GetAtoms():
                atom_name = atom.GetName().strip()
                if atom_name == "C":
                    structure.DeleteAtom(atom)
        structure = delete_partial_residues(structure)
        assert get_sequence(structure) == sequence


@pytest.mark.parametrize(
    "package, resource, sequence",
    [
        (  # delete last three residues
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            "MNTFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTV",
        ),
    ],
)
def test_delete_short_protein_segments(package, resource, sequence):
    """Compare results to expected sequence."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        structure = delete_short_protein_segments(structure)
        assert get_sequence(structure) == sequence


@pytest.mark.parametrize(
    "package, resource, cutoff, sequence",
    [
        (  # PHE4 is clashing (< 2.0) with ASP33 and GLU112
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            2.0,
            "THVDFAPNTGEIFAGKQPGDVTMFTLTMGTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEYVFDNKQSTVINSKDVSGEVTVQG",
        ),
        (  # setting the cutoff very low should not delete PHE4, ASP33 and GLU112
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            0.1,
            "TFHVDFAPNTGEIFAGKQPGDVTMFTLTMGDTAPHGGWRLIPTGDSKGGYMISADGDYVGLYSYMMSWVGIDNNWYINDSPKDIKDHLYVKAGTVLKPTTYKFTGRVEEYVFDNKQSTVINSKDVSGEVTVQG",
        ),
        (  # setting the cutoff very high should delete all sidechains, except for glycines
            "kinoml.data.proteins",
            "4f8o_edit.pdb",
            4.0,
            "GGGGGGGGGGGGGGGG",
        ),
    ],
)
def test_delete_clashing_sidechains(package, resource, cutoff, sequence):
    """Compare results to expected sequence."""
    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        structure = delete_clashing_sidechains(structure, cutoff)
        structure = delete_partial_residues(structure)
        assert get_sequence(structure) == sequence


@pytest.mark.parametrize(
    "package, resource",
    [
        (
            "kinoml.data.proteins",
            "4f8o.pdb",
        ),
        (
            "kinoml.data.molecules",
            "chloroform.sdf",
        ),
    ],
)
def test_get_atom_coordinates(package, resource):
    """
    Compare results to have the same number of coordinates as atoms and to have exactly three
    floats as coordinates.
    """
    import itertools

    with resources.path(package, resource) as path:
        structure = read_molecules(str(path))[0]
        coordinates = get_atom_coordinates(structure)
        all_floats = all(
            [
                isinstance(coordinate, float) for coordinate
                in itertools.chain.from_iterable(coordinates)
            ]
        )
        assert structure.NumAtoms() == len(coordinates)
        assert set([len(coordinate) for coordinate in coordinates]) == {3}
        assert all_floats
