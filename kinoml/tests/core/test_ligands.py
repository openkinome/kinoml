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
    assert isinstance(ligand.rdkit_mol, rdkit.Chem.Mol)
    assert isinstance(ligand.openff_mol, Molecule)
    assert isinstance(ligand.openeye_mol, oechem.OEGraphMol)
    assert isinstance(ligand.smiles, str)
    assert isinstance(ligand.get_canonical_smiles(), str)
