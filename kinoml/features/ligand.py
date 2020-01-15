"""
Featurizers built around ``kinoml.core.ligand.Ligand`` objects.
"""

import numpy as np

from .base import _BaseFeaturizer
from .utils import one_hot_encode


class MorganFingerprintFeaturizer(_BaseFeaturizer):

    """
    Featurizes a ``Ligand`` using Morgan fingerprints bitvectors

    Parameters
    ==========
    molecule : kinoml.core.ligand.RDKitMolecule
    radius : int, optional=2
        Morgan fingerprint neighborhood radius
    nbits : int, optional=512
        Length of the resulting bit vector
    """

    def __init__(self, molecule, radius=2, nbits=512, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.radius = radius
        self.nbits = nbits

    def _featurize(self):
        """
        Featurizes ``self.molecule`` as a Morgan Fingerprint using RDKit

        Returns
        ========
        np.array
            Morgan fingerprint of radius ``radius`` of molecule, with shape ``nbits``.
        """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        m = self.molecule
        if m is not None:
            return np.array(GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.nbits))



class OneHotSMILESFeaturizer(_BaseFeaturizer):

    """
    One-hot encodes a ``Ligand`` from a canonical SMILES representation.

    Parameters
    ==========
    molecule : kinoml.core.ligand.RDKitMolecule
    pad_up_to : int or None, optional=None
        Fill featurized data with zeros until pad-length is met.

    Attributes
    ==========
    DICTIONARY : dict
        Defines the character-integer mapping of the one-hot encoding.
    """

    DICTIONARY = {c: i for i, c in enumerate(
        'BCFHIKNOPSUVWY'  # atoms
        'acegilnosru'  # aromatic atoms
        '-=#'  # bonds
        '1234567890'  # ring closures
        '.*'  # disconnections
        '()'  # branches
        '/+@:[]%\\'  # other characters
        'LR$'  # single-char representation of Cl, Br, @@
    )}

    def __init__(self, molecule, pad_up_to=None, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.pad_up_to = pad_up_to

    def _featurize(self):
        """
        Featurizes ``self.molecule`` as a one-hot encoding of the SMILES representation.
        If ``self.pad_up_to`` is defined, the padded version will be returned.

        Returns
        =======
        np.array
            One-hot encoded SMILES, with shape (``len(self.DICTIONARY``, ``self.pad_up_to``, )).

        Notes
        =======
        Double element symbols (such as Cl, Br for atoms and @@ for chirality) are replaced
        with single element symbols (L, R and $ respectively).
        """
        smiles = self.molecule.to_smiles().replace("Cl", "L").replace("Br", "R").replace("@@", "$")
        ohe_matrix = one_hot_encode(smiles, self.DICTIONARY)

        if self.pad_up_to is not None:
            return np.pad(ohe_matrix, ((0,0), (0, self.pad_up_to-len(smiles))), mode='constant')
        return ohe_matrix

     

class GraphFeaturizer(_BaseFeaturizer):

    """
    Creates a graph representation of ``self.molecule``, with ``N_FEATURES``
    features per atom. Check ``self._features_per_atom`` for details.

    Parameters
    ==========
    molecule : kinoml.core.ligand.RDKitMolecule
    pad_up_to : int or None, optional=None
        Fill featurized data with zeros until pad-length is met.
    """

    N_FEATURES = 2

    def __init__(self, molecule, pad_up_to=None, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.pad_up_to = pad_up_to

    def _featurize(self):
        """
        Featurizes ``self.molecule`` as a labeled graph.
        If ``self.pad_up_to`` is defined, the padded version will be returned.

        Returns
        =======
        np.array
            Graph matrix, with shape (``self.pad_up_to``, ``self.N_FEATURES``)).

        Notes
        ======
        Please refer to https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
        for the Pytorch implementation.
        Please refer to https://arxiv.org/pdf/1609.02907.pdf for the original article.
        """

        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdmolops
        from scipy.linalg import fractional_matrix_power

        mol = self.molecule

        self_adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol) + np.identity(mol.GetNumAtoms())
        per_atom_features = np.matrix([self._per_atom_features(atom) for atom in mol.GetAtoms()])
        degree_matrix = np.matrix(np.diag([a.GetDegree() for a in mol.GetAtoms()]))
        self_degree_matrix = np.matrix(np.diag([a.GetDegree() + 1 for a in mol.GetAtoms()]))
        inv_self_degree_matrix = fractional_matrix_power(self_degree_matrix, -0.5)

        out = inv_self_degree_matrix * self_adjacency_matrix * inv_self_degree_matrix * per_atom_features

        if self.pad_up_to is not None:
            out = np.pad(out, (0, self.pad_up_to - out.shape[0]))

        return out

    def _per_atom_features(self, atom):
        """
        Computes desired features for each atom in the graph.
        If you subclass this method, remember to update ``self.N_FEATURES`` too.

        Parameters
        ==========
        atom : rdkit.Chem.Atom

        Returns
        =======
        tuple
            Atomic number, number of neighbors
        """
        return atom.GetAtomicNum(), atom.GetDegree()
