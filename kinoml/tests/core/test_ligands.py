"""
Test kinoml.core.ligands
"""


def test_ligand():
    from kinoml.core.ligands import Ligand, BaseLigand

    smiles = "CCCCC"
    ligand = Ligand.from_smiles(smiles)
    assert isinstance(ligand, BaseLigand)
    assert ligand._provenance["smiles"] == smiles != ligand.to_smiles()
