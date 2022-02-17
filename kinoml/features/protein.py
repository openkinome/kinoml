"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
from collections import Counter
import logging
from typing import Union

import numpy as np

from .core import ParallelBaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.proteins import Protein, KLIFSKinase
from ..core.sequences import AminoAcidSequence
from ..core.systems import ProteinSystem, ProteinLigandComplex


logger = logging.getLogger(__name__)


class SingleProteinFeaturizer(ParallelBaseFeaturizer):
    """
    Provides a minimally useful ``._supports()`` method for all Protein-like featurizers.
    """

    _COMPATIBLE_LIGAND_TYPES = (Protein, KLIFSKinase)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _supports(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> bool:
        """
        Check that exactly one protein is present in the System
        """
        super_checks = super()._supports(system)
        proteins = [c for c in system.components if isinstance(c, self._COMPATIBLE_LIGAND_TYPES)]
        return all([super_checks, len(proteins) == 1])


class AminoAcidCompositionFeaturizer(SingleProteinFeaturizer):

    """Featurizes the protein using the composition of the residues in the binding site."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] = 0

    def _featurize_one(
            self, system: Union[ProteinSystem, ProteinLigandComplex]
    ) -> Union[np.array, None]:
        """
        Featurizes a protein using the residue count in the sequence.

        Parameters
        ----------
        system: ProteinSystem or ProteinLigandComplex
            The System to be featurized.

        Returns
        -------
        : np.array or None
            The count of amino acids in the binding site.
        """
        count = self._counter.copy()
        try:
            sequence = system.protein.sequence
        except ValueError:  # e.g. erroneous uniprot_id in lazy instantiation
            return None
        count.update(system.protein.sequence)
        sorted_count = sorted(count.items(), key=lambda kv: kv[0])
        return np.array([number for _, number in sorted_count])


class OneHotEncodedSequenceFeaturizer(BaseOneHotEncodingFeaturizer, SingleProteinFeaturizer):

    """Featurizes the sequence of the protein to a one hot encoding."""

    ALPHABET = AminoAcidSequence.ALPHABET

    def __init__(self, sequence_type: str = "full", **kwargs):
        """
        Featurizes the sequence of the protein to a one hot encoding.

        Parameters
        ----------
        sequence_type: str, default=full
            The sequence to use for one hot encoding ('full', 'klifs_kinase' or 'klifs_structure').
        """
        super().__init__(**kwargs)
        if sequence_type not in ["full", "klifs_kinase", "klifs_structure"]:
            raise ValueError(
                "Only 'full', 'klifs_kinase' and 'klifs_structure' are supported sequence_types, "
                f"you provided {sequence_type}."
            )
        self.sequence_type = sequence_type

    def _retrieve_sequence(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> str:
        try:
            if self.sequence_type == "full":
                sequence = system.protein.sequence
            elif self.sequence_type == "klifs_kinase":
                sequence = system.protein.kinase_klifs_sequence
            else:
                sequence = system.protein.structure_klifs_sequence
        except ValueError:  # e.g. erroneous uniprot_id in lazy instantiation
            return ""
        return sequence
