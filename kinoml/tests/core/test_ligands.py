"""
Test kinoml.core.ligands
"""


def test_ligand():
    from openeye import oechem
    from openff.toolkit.topology import Molecule
    import rdkit

    from kinoml.core.ligands import Ligand
    from kinoml.core.components import BaseLigand

    smiles = "CCCCC"
    ligand = Ligand(smiles=smiles)
    assert isinstance(ligand, BaseLigand)
    assert isinstance(ligand.to_rdkit(), rdkit.Chem.Mol)
    assert isinstance(ligand.to_openff(), Molecule)
    assert isinstance(ligand.to_openeye(), oechem.OEGraphMol)
    assert isinstance(ligand.to_smiles(), str)
