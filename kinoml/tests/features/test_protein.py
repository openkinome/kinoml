"""
Test ligand featurizers of `kinoml.protein`
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
            216
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o.pdb"],
            None,
            None,
            None,
            None,
            None,
            239
        ),
        (
            "kinoml.data.proteins",
            ["kinoml_tests_4f8o_spruce.loop_db", "4f8o_edit.pdb"],
            None,
            "P31522",
            None,
            None,
            None,
            109
        ),
    ],
)
def test_OEProteinStructureFeaturizer(
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
    from kinoml.core.proteins import BaseProtein
    from kinoml.core.systems import ProteinSystem
    from kinoml.features.protein import OEProteinStructureFeaturizer

    with resources.path(package, resource_list[0]) as loop_db:
        featurizer = OEProteinStructureFeaturizer(loop_db=loop_db, use_multiprocessing=False)
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
            system = ProteinSystem([base_protein])
            system = featurizer.featurize([system])[0]
            u = system.featurizations["OEProteinStructureFeaturizer"].universe
            n_residues = u.residues.n_residues
            assert n_residues == expected_n_residues
        else:
            with resources.path(package, resource_list[1]) as path:
                base_protein.path = path
                system = ProteinSystem([base_protein])
                system = featurizer.featurize([system])[0]
                u = system.featurizations["OEProteinStructureFeaturizer"].universe
                n_residues = u.residues.n_residues
                assert n_residues == expected_n_residues
