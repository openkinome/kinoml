"""
Featurization utilities
"""

import numpy as np


def pad(data, shape, fill=0):
    """
    Given an array-like data object, pad it with ``fill`` up to shape ``shape``
    """
    pass


def one_hot_encode(sequence, dictionary):
    """
    One-hot encode a sequence of characters, given a dictionary

    Parameters
    ==========
    sequence : str
    dictionary : dict
        Mapping of each character to their position in the alphabet

    Returns
    =======
    np.array
        One-hot encoded matrix with shape (len(dictionary), len(sequence))
    """
    ohe_matrix = np.zeros((len(dictionary), len(sequence)))
    for i, character in enumerate(sequence):
        ohe_matrix[dictionary[character],i] = 1
    return ohe_matrix
