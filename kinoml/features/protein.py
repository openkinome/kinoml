"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from collections import Counter

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.proteins import BaseProtein, AminoAcidSequence


class AminoAcidCompositionFeaturizer(BaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues
    in the binding site.
    """

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] -= 1

    def _featurize(self) -> np.array:
        """
        Featurizes a protein using the residue count in the sequence

        Returns:
            The count of amino acid in the binding site.
        """
        count = self._counter.copy()
        count.update(self.molecule.sequence)
        return np.array(list(count.values()))


class OneHotEncodedSequenceFeaturizer(BaseOneHotEncodingFeaturizer):

    """
    Featurize the sequence of the protein to a one hot encoding
    using the symbols in ``ALL_AMINOACIDS``.
    """

    ALPHABET = AminoAcidSequence.ALPHABET

    def _retrieve_sequence(self, system: System) -> str:
        for comp in system.components:
            if isinstance(comp, AminoAcidSequence):
                return comp.sequence
