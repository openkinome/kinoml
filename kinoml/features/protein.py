"""
Featurizers built around ``kinoml.core.protein.Protein`` objects.
"""

from collections import Counter

import numpy as np
#import tensorflow as tf

from .base import _BaseFeaturizer


ALL_AMINOACIDS = "ACDEFGHIKLMNPQRSTVWY"


class HashFeaturizer(_BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.
    """

    def _featurize(self):
        # write the featurization strategy here
        return hash(self.molecule.id)


class AminoAcidCompositionFeaturizer(_BaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues in the binding site.
    """

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(ALL_AMINOACIDS))
    for k in _counter:
        _counter[k] -= 1

    def _featurize(self):
        """
        Featurizes a protein using the residue count in the binding site.

        Returns
        =======
        np.array
            The count of amino acid in the binding site, with shape (``len(ALL_AMINOACIDS``)).

        """
        count = self._counter.copy()
        count.update(self.molecule.sequence)
        return np.array(count.values())


class SequenceFeaturizer(_BaseFeaturizer):

    """
    Featurize the sequence of the protein to a one hot encoding
    using the symbols in ``ALL_AMINOACIDS``.

    """

    DICTIONARY = {c: i for i, c in enumerate(ALL_AMINOACIDS)}

    def __init__(self, pad_up_to=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_up_to = pad_up_to

    def _featurize(self):
        """
        Featurizes the binding site sequence of a protein using
        a one hot encoding of the amino acids.

        Returns
        =======
        np.matrix
            One hot encoding of the sequence, with shape (``len(ALL_AMINOACIDS)``, ``len(self.molecule.sequence)``).

        """
        vec = np.array([self.DICTIONARY[aa] for aa in self.molecule.sequence])
        ohe = tf.keras.utils.to_categorical(vec, len(self.DICTIONARY))

        if self.pad_up_to is not None:
            ohe = np.pad(vec, (0, self.pad_up_to - len(ohe)))

        return ohe


class OneHotEncodingAminoAcid:
    """
    Featurizes the binding site sequence of a protein using
    a one hot encoding of the amino acids.
    This class assumes that the sequence of the binding site is composed of 20 amino acids (as is the case in KLIFS)
    and therefore does not require padding.

    Returns
    =======
    np.matrix
        One hot encoding of the sequence, with shape (``len(ALL_AMINOACIDS)``, ``len(self.molecule.sequence)``).
    """

    def __init__(self, amino_acid_seq):
        self.amino_acid_seq = amino_acid_seq
        self.len_seq = len(amino_acid_seq)

    def from_seq2oh(self):
        ohe_matrix = np.zeros((len(ALL_AMINOACIDS), self.len_seq))
        for i, dict_char in enumerate(ALL_AMINOACIDS):
            for j, seq_char in enumerate(self.amino_acid_seq):
                if dict_char == seq_char:
                    ohe_matrix[i,j] = 1
        return ohe_matrix
