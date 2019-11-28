"""
Featurizers built around ``kinoml.core.ligand.Ligand`` objects.
"""

import numpy as np
import tensorflow as tf
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from .base import _BaseFeaturizer


class OneHotFeaturizer(_BaseFeaturizer):

    """
    One-hot encodes a ``Ligand``'s canonical SMILES representation

    Parameters
    ==========
    pad : int or None, optional=None
        Fill featurized data with zeros until pad-length is met
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

    def __init__(self, pad=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = pad

    def _featurize(self):
        smiles = self.molecule.to_smiles().replace("Cl", "L").replace("Br", "R").replace("@@", "$")
        vec = np.array([self.DICTIONARY[c] for c in smiles])
        if self.pad is not None:
            vec = np.pad(vec, (0, self.pad - len(vec)))
        return tf.keras.utils.to_categorical(vec, len(self.DICTIONARY))


class MorganFingerprintFeaturizer(_BaseFeaturizer):

    """
    Featurizes a ``Ligand`` using Morgan fingerprints bitvectors

    Parameters
    ==========
    radius : int, optional=2
        Morgan fingerprint neighborhood radius
    nbits : int, optional=512
        Length of the resulting bit vector
    """

    def __init__(self, radius=2, nbits=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.nbits = nbits

    def _featurize(self):
        m = self.molecule.to_rdkit()
        if m is None:
            return np.nan
        return np.array(GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.nbits))
