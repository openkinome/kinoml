"""
Featurizers built around ``kinoml.core.protein.Protein`` objects.
"""

import numpy as np

from .base import _BaseFeaturizer


class HashFeaturizer(_BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.
    """

    def _featurize(self):
        # write the featurization strategy here
        return hash(self.molecule.id)


class SequenceFeaturizer(_BaseFeaturizer):

    """
    Featurize the sequence of the protein to a int-vector

    """

    DICTIONARY = {c: i for i, c in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    def _featurize(self):
        sequence = self.molecule.sequence
        return np.array([self.DICTIONARY[aa] for aa in sequence])
