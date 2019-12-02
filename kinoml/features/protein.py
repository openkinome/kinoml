"""
Featurizers built around ``kinoml.core.protein.Protein`` objects.
"""

from collections import Counter

import numpy as np
import tensorflow as tf

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

    """

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(ALL_AMINOACIDS))
    for k in _counter:
        _counter[k] -= 1

    def _featurize(self):
        count = _counter.copy()
        count.update(self.molecule.sequence)
        return np.array(count.values())


class SequenceFeaturizer(_BaseFeaturizer):

    """
    Featurize the sequence of the protein to a int-vector

    """

    DICTIONARY = {c: i for i, c in enumerate(ALL_AMINOACIDS)}

    def __init__(self, pad_up_to=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_up_to = pad_up_to

    def _featurize(self):
        vec = np.array([self.DICTIONARY[aa] for aa in self.molecule.sequence])
        ohe = tf.keras.utils.to_categorical(vec, len(self.DICTIONARY))

        if self.pad_up_to is not None:
            ohe = np.pad(vec, (0, self.pad_up_to - len(ohe)))

        return ohe
