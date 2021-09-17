"""
Test ligand featurizers of `kinoml.complexes`
"""
from importlib import resources
import pytest


@pytest.mark.parametrize(
    "package, resource_list, pdb_id, uniprot_id, chain_id, alternate_location, expo_id, "
    "expected_n_residues",
    [
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db"],
            "4f8o",
            "P31522",
            "A",
            "B",
            "AES",
            217
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.pdb"],
            None,
            None,
            None,
            None,
            None,
            240
        ),
    ],
)
def test_OEComplexFeaturizer(
        package,
        resource_list,
        pdb_id,
        uniprot_id,
        chain_id,
        alternate_location,
        expo_id,
        expected_n_residues
):
    """
    Compare featurizer results to expected number of residues.
    """
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEComplexFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEComplexFeaturizer(loop_db=loop_db)
        ligand = Ligand(name="LIG")
        base_protein = BaseProtein(name="PsaA")
        if uniprot_id:
            base_protein.uniprot_id = uniprot_id
        if chain_id:
            base_protein.chain_id = chain_id
        if alternate_location:
            base_protein.alternate_location = alternate_location
        if expo_id:
            base_protein.expo_id = expo_id
        if pdb_id:
            base_protein.pdb_id = pdb_id
            system = ProteinLigandComplex([base_protein, ligand])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEComplexFeaturizer"].universe
            n_residues = u.residues.n_residues
            assert n_residues == expected_n_residues
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinLigandComplex([base_protein, ligand])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEComplexFeaturizer"].universe
                n_residues = u.residues.n_residues
                assert n_residues == expected_n_residues


@pytest.mark.parametrize(
    "package, resource_list, pdb_id, uniprot_id, chain_id, alternate_location, expo_id, "
    "smiles, expected_n_ligand_atoms",
    [
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db"],
            "4f8o",
            "P31522",
            "A",
            "B",
            "AES",
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.pdb"],
            None,
            None,
            None,
            None,
            None,
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.cif"],
            None,
            None,
            None,
            None,
            None,
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
    ],
)
def test_OEHybridDockingFeaturizer(
        package,
        resource_list,
        pdb_id,
        uniprot_id,
        chain_id,
        alternate_location,
        expo_id,
        smiles,
        expected_n_ligand_atoms
):
    """
    Compare featurizer results to expected number of ligand atoms.
    """
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEHybridDockingFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEHybridDockingFeaturizer(loop_db=loop_db)
        ligand = Ligand.from_smiles(smiles=smiles, name="LIG")
        base_protein = BaseProtein(name="PsaA")
        if uniprot_id:
            base_protein.uniprot_id = uniprot_id
        if chain_id:
            base_protein.chain_id = chain_id
        if alternate_location:
            base_protein.alternate_location = alternate_location
        if expo_id:
            base_protein.expo_id = expo_id
        if pdb_id:
            base_protein.pdb_id = pdb_id
            system = ProteinLigandComplex([base_protein, ligand])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEHybridDockingFeaturizer"].universe
            n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
            assert n_ligand_atoms == expected_n_ligand_atoms
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinLigandComplex([base_protein, ligand])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEHybridDockingFeaturizer"].universe
                n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
                assert n_ligand_atoms == expected_n_ligand_atoms


@pytest.mark.parametrize(
    "package, resource_list, pdb_id, uniprot_id, chain_id, alternate_location, expo_id, "
    "pocket_resids, smiles, expected_n_ligand_atoms",
    [
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db"],
            "4f8o",
            None,
            "A",
            "B",
            "AES",
            [50, 51, 52, 62, 63, 64, 70, 77],
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o_edit.pdb"],
            None,
            "P31522",
            None,
            None,
            None,
            [50, 51, 52, 62, 63, 64, 70, 77],
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
    ],
)
def test_OEFredDockingFeaturizer(
        package,
        resource_list,
        pdb_id,
        uniprot_id,
        chain_id,
        alternate_location,
        expo_id,
        pocket_resids,
        smiles,
        expected_n_ligand_atoms
):
    """
    Compare featurizer results to expected number of ligand atoms.
    """
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEFredDockingFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEFredDockingFeaturizer(loop_db=loop_db)
        ligand = Ligand.from_smiles(smiles=smiles, name="LIG")
        base_protein = BaseProtein(name="PsaA")
        base_protein.pocket_resids = pocket_resids
        if uniprot_id:
            base_protein.uniprot_id = uniprot_id
        if chain_id:
            base_protein.chain_id = chain_id
        if alternate_location:
            base_protein.alternate_location = alternate_location
        if expo_id:
            base_protein.expo_id = expo_id
        if pdb_id:
            base_protein.pdb_id = pdb_id
            system = ProteinLigandComplex([base_protein, ligand])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEFredDockingFeaturizer"].universe
            n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
            assert n_ligand_atoms == expected_n_ligand_atoms
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinLigandComplex([base_protein, ligand])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEFredDockingFeaturizer"].universe
                n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
                assert n_ligand_atoms == expected_n_ligand_atoms


@pytest.mark.parametrize(
    "package, resource_list, pdb_id, uniprot_id, chain_id, alternate_location, expo_id, "
    "smiles, expected_n_ligand_atoms",
    [
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db"],
            "4f8o",
            "P31522",
            "A",
            "B",
            "AES",
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.pdb"],
            None,
            None,
            None,
            None,
            None,
            "c1cc(ccc1CCN)S(=O)(=O)F",
            24
        ),
    ],
)
def test_OEPositDockingFeaturizer(
        package,
        resource_list,
        pdb_id,
        uniprot_id,
        chain_id,
        alternate_location,
        expo_id,
        smiles,
        expected_n_ligand_atoms
):
    """
    Compare featurizer results to expected number of ligand atoms.
    """
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEPositDockingFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEPositDockingFeaturizer(loop_db=loop_db)
        ligand = Ligand.from_smiles(smiles=smiles, name="LIG")
        base_protein = BaseProtein(name="PsaA")
        if uniprot_id:
            base_protein.uniprot_id = uniprot_id
        if chain_id:
            base_protein.chain_id = chain_id
        if alternate_location:
            base_protein.alternate_location = alternate_location
        if expo_id:
            base_protein.expo_id = expo_id
        if pdb_id:
            base_protein.pdb_id = pdb_id
            system = ProteinLigandComplex([base_protein, ligand])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEPositDockingFeaturizer"].universe
            n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
            assert n_ligand_atoms == expected_n_ligand_atoms
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinLigandComplex([base_protein, ligand])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEPositDockingFeaturizer"].universe
                n_ligand_atoms = u.select_atoms("resname LIG").atoms.n_atoms
                assert n_ligand_atoms == expected_n_ligand_atoms
