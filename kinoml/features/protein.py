"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from .core import ParallelBaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.proteins import AminoAcidSequence


class AminoAcidCompositionFeaturizer(ParallelBaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues from the sequence of the protein.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] = 0

    def _featurize_one(self, system: System) -> np.array:
        """
        Featurizes a protein using the residue count in the sequence.

        Parameters
        ----------
        system: System
            The System to be featurized. Sometimes it will

        Returns
        -------
        list of array
            The count of amino acid in the binding site.
        """
        count = self._counter.copy()
        count.update(system.protein.sequence)
        sorted_count = sorted(count.items(), key=lambda kv: kv[0])
        return np.array([number for _, number in sorted_count])


class OneHotEncodedSequenceFeaturizer(BaseOneHotEncodingFeaturizer):

    """
    Featurizes the protein using the one-hot encoded sequence, using the characters in ``ALPHABET``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ALPHABET = AminoAcidSequence.ALPHABET

    def _retrieve_sequence(self, system: System) -> str:
        for comp in system.components:
            if isinstance(comp, AminoAcidSequence):
                return comp.sequence
