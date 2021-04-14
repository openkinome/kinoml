"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.proteins import AminoAcidSequence
from ..datasets.core import DatasetProvider


class AminoAcidCompositionFeaturizer(BaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues
    in the binding site.
    """

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] = 0

    def _featurize(
        self, system: System, dataset: DatasetProvider, inplace: bool = True
    ) -> np.array:
        """
        Featurizes a protein using the residue count in the sequence

        Parameters
        ----------
        system: System
            The System to be featurized. Sometimes it will
        dataset : DatasetProvider
            The full DatasetProvider which the System belongs to. Useful
            if the featurizer needs to compute a global property (e.g.
            one-hot encoding needs the maximum length)
        inplace: bool, optional
            Whether to modify the System directly or operate on a copy.

        Returns
        -------
        array
            The count of amino acid in the binding site.
        """
        count = self._counter.copy()
        count.update(system.protein.sequence)
        sorted_count = sorted(count.items(), key=lambda kv: kv[0])
        return np.array([number for aminoacid, number in sorted_count])


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
