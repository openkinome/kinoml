"""
Test kinoml.core.ligands
"""
from importlib import resources


def test_ligand():
    from openeye import oechem
    from openff.toolkit.topology import Molecule
    import rdkit

    from kinoml.core.ligands import Ligand
    from kinoml.core.components import BaseLigand

    smiles = "CCCCC"
    ligand = Ligand.from_smiles(smiles=smiles)
    assert isinstance(ligand.molecule, Molecule)
    with resources.path("kinoml.data.molecules", "chloroform.sdf") as path:
        ligand = Ligand.from_file(str(path))
        assert isinstance(ligand.molecule, Molecule)
    ligand = Ligand(smiles=smiles)
    assert isinstance(ligand, BaseLigand)
    assert isinstance(ligand.molecule, Molecule)
    assert isinstance(ligand.molecule.to_rdkit(), rdkit.Chem.Mol)
    assert isinstance(ligand.molecule.to_openeye(), oechem.OEMol)
    assert isinstance(ligand._smiles, str)
    assert isinstance(ligand.metadata["smiles"], str)
    assert isinstance(ligand.molecule.to_smiles(), str)
