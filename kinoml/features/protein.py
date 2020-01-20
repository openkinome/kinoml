"""
Featurizers built around ``kinoml.core.protein.Protein`` objects.
"""

from collections import Counter

import numpy as np
import hashlib

from .base import _BaseFeaturizer
from .utils import one_hot_encode

ALL_AMINOACIDS = "ACDEFGHIKLMNPQRSTVWY"


class HashFeaturizer(_BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.
    
    Parameters
    ==========
    molecule : kinoml.core.protein
    normalize : bool, default=True
        Normalizes the hash to obtain a value in the unit interval
    """

    def __init__(self, molecule, normalize=True, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.molecule = molecule
        self.normalize = normalize

    def _featurize(self):
        """
        Featurizes the protein using the hash of the protein name.

        Returns
        =======
        The hash of the name of the molecule,
        normalized by default (value between 0 and 1)

        """
        h = hashlib.sha256(self.molecule.name.encode(encoding='UTF-8'))
        if self.normalize:
            return int(h.hexdigest(), base=16)/2**256
        return int(h.hexdigest(), base=16)
        


class AminoAcidCompositionFeaturizer(_BaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues
    in the binding site.
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
            The count of amino acid in the binding site
            with shape (``len(ALL_AMINOACIDS``)).

        """
        count = self._counter.copy()
        count.update(self.molecule.sequence)
        return np.array(list(count.values()))


class SequenceFeaturizer(_BaseFeaturizer):

    """
    Featurize the sequence of the protein to a one hot encoding
    using the symbols in ``ALL_AMINOACIDS``.

    """

    DICTIONARY = {c: i for i, c in enumerate(ALL_AMINOACIDS)}

    def __init__(self, molecule, pad_up_to=None, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.pad_up_to = pad_up_to
    

    def _featurize(self):
        """
        Featurizes the binding site sequence of a protein using
        a one hot encoding of the amino acids.
        If ``self.pad_up_to`` is defined, the padded version will be returned.

        Returns
        =======
        np.matrix
            One hot encoding of the sequence with
            shape (``len(ALL_AMINOACIDS)``, ``len(self.molecule.sequence)``).

        """
        ohe_matrix = one_hot_encode(self.molecule.sequence, self.DICTIONARY)
        if self.pad_up_to is not None:
            return np.pad(ohe_matrix,
            ((0,0), (0, self.pad_up_to-len(self.molecule.sequence))),
            mode='constant')
        return ohe_matrix


class GraphSeqFeaturizer(_BaseFeaturizer):
    """
    Creates a graph representation of the sequence of the protein,
    using the symbols in ``ALL_AMINOACIDS``.
    """
    
    DICTIONARY = {c: i for i, c in enumerate(ALL_AMINOACIDS)}

    def __init__(self, molecule, pad_up_to=None, *args, **kwargs):
        super().__init__(molecule, *args, **kwargs)
        self.molecule = molecule
        self.pad_up_to = pad_up_to

    def _featurize(self):
        """
        Featurize the sequence of the protein as a graph:
        the adjacency matrix which is a tridiagonal matrix and per node feature, 
        which is the one-hot encoding of the amino acids, using ``DICTIONARY``
        If ``self.pad_up_to`` is defined, the padded version will be returned.

        Returns
        =======
        tuple of 2 elements
            - Adjacency matrix of the amino acid sequence (N_amino_acid, N_amino_acid),
            where N_amino_acid is the number of amino acids.
        - Feature matrix with shape (N_amino_acid, ``len(self.DICTIONARY)``)
        """

        adjacency_matrix = np.diag([1]*(len(self.molecule.sequence)-1), 1) + np.diag([1]*(len(self.molecule.sequence)-1), -1)
        per_node_feature_matrix = one_hot_encode(self.molecule.sequence, self.DICTIONARY)

        if self.pad_up_to is not None:
            adjacency_matrix_pad = np.pad(adjacency_matrix,
            ((0, self.pad_up_to - len(self.molecule.sequence)),
            (0, self.pad_up_to - len(self.molecule.sequence))),
            mode='constant')
            per_node_feature_pad = np.pad(per_node_feature_matrix,
            ((0, 0), (0,self.pad_up_to - len(self.molecule.sequence))),
            mode='constant')
            return (adjacency_matrix_pad, per_node_feature_pad)
        return (adjacency_matrix, per_node_feature_matrix)