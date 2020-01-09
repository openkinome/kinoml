"""
Core objects to deal with ligands and small compounds
"""

from openforcefield.topology import Molecule
from rdkit.Chem import MolFromSmiles, MolToSmiles

class Ligand(Molecule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RDKitLigand(object):

    """
    A Ligand object based on RDKit Molecules

    Parameters
    ----------
    rdmol : rdkit.Chem.Molecule
    """
    def __init__(self, rdmol):
        self.molecule = rdmol

    @classmethod
    def from_smiles(cls, smiles):
        m = MolFromSmiles(smiles)
        return cls(m)

    def to_smiles(self):
        return MolToSmiles(self.molecule)
