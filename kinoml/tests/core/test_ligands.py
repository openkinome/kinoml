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
    assert isinstance(ligand.molecule, Molecule)
    assert isinstance(ligand.molecule.to_rdkit(), rdkit.Chem.Mol)
    assert isinstance(ligand.molecule.to_openeye(), oechem.OEMol)
    assert isinstance(ligand._smiles, type(None))
    assert isinstance(ligand.metadata["smiles"], str)
    assert isinstance(ligand.molecule.to_smiles(), str)
