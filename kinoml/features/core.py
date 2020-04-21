"""
Featurizers can transform a `kinoml.core.system.System` object and produce
new representations of the molecular entities and their associated measurements.

All `Featurizer` objects inherit from `BaseFeaturizer` and reimplement `._featurize`
and `._supports`, if needed.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Hashable

import numpy as np

from ..core.systems import System


class BaseFeaturizer:
    """
    Abstract Featurizer class
    """

    _SUPPORTED_TYPES = (System,)

    def featurize(self, system, inplace: bool = False) -> object:
        """
        Given an system (compatible with `_SUPPORTED_TYPES`), apply
        the featurization scheme implemented in this class.

        Parameters:
            data: This is the data system to be transformed
            inplace: Whether to modify the system directly or operate on a copy.
        """
        if not inplace:
            system = deepcopy(system)
        self.supports(system)
        return self._featurize(system)

    def _featurize(self, system: System) -> object:
        """
        Implement this method to do the actual work for `self.featurize()`.
        """
        return None

    def supports(self, *systems: System, raise_errors: bool = True) -> bool:
        """
        Check if these systems are supported by this featurizer.

        Do NOT reimplement in subclass. Check `._supports()` instead.

        Parameters:
            systems: systems to be checked (by type, contained attributes, etc)
            raise_errors: if True, raise `ValueError`.

        Returns:
            True if all systems are compatible, False otherwise

        Raises:
            `ValueError` if `._supports()` fails and `raise_errors` is `True`.
        """
        for system in systems:
            if not self._supports(system):
                if raise_errors:
                    raise ValueError(f"{self.name} does not support {system}")
                return False
        return True

    def _supports(self, system: System) -> bool:
        """
        This is the private method that actually tests for compatibility between
        a single system and the current featurizer.

        This is the method you should reimplement in your subclass.

        Parameters:
            system: the system that will be checked

        Returns:
            True if compatible, False otherwise
        """
        return isinstance(system, self._SUPPORTED_TYPES)

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"<{self.name}>"


class Stacked(BaseFeaturizer):

    """
    Featurizer-compatible class that provides a way to stack
    featurizers together

    Parameters:
        featurizers: featurizers to stack. They must be compatible!
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer]):
        self.featurizers = featurizers

    def _featurize(self, system):
        for featurizer in self.featurizers:
            system = featurizer.featurize(system, inplace=True)
        return system

    def supports(self, *systems: System, raise_errors: bool = False) -> bool:
        """
        Check if these systems are supported by all featurizers.

        Parameters:
            systems: systems to be checked (by type, contained attributes, etc)
            raise_errors: if True, raise `ValueError`.

        Returns:
            True if all systems are compatible with all featurizers, False otherwise

        Raises:
            `ValueError` if `f.supports()` fails and `raise_errors` is `True`.
        """
        return all(
            f.supports(s, raise_errors=raise_errors) for f in self.featurizers for s in systems
        )

    @property
    def name(self):
        return f"{self.__class__.__name__}([{', '.join([f.__name__.__class__ for f in self.featurizers])}])"


class BaseOneHotEncodingFeaturizer(BaseFeaturizer):
    ALPHABET = None

    def __init__(self, dictionary: dict = None):
        if dictionary is None:
            dictionary = {c: i for i, c in enumerate(self.ALPHABET)}
        self.dictionary = dictionary
        if not self.dictionary:
            raise ValueError("This featurizer requires a populated dictionary!")

    def _featurize(self, system: System):
        sequence = self._retrieve_sequence(system)
        return self.one_hot_encode(sequence, self.dictionary)

    def _retrieve_sequence(self, system: System):
        """
        Implement in your component-specific subclass!
        """
        raise NotImplementedError

    @staticmethod
    def one_hot_encode(sequence: str, dictionary: dict) -> np.ndarray:
        """
        One-hot encode a sequence of characters, given a dictionary

        Parameters:
            sequence : str
            dictionary: Mapping of each character to their position in the alphabet

        Returns:
            One-hot encoded matrix with shape (len(dictionary), len(sequence))
        """
        ohe_matrix = np.zeros((len(dictionary), len(sequence)))
        for i, character in enumerate(sequence):
            ohe_matrix[dictionary[character], i] = 1
        return ohe_matrix


class PadFeaturizer(BaseFeaturizer):
    """
    Pads features of a given system to a desired size or length.

    This class wraps `numpy.pad` with `mode=constant`, auto-calculating
    the needed additions to match the requested shape.

    Parameters:
        shape: the desired size of the transformed features
        key: element to retrieve from `System.featurizations`
        pad_with: value to fill the array-like features with
    """

    def __init__(self, shape: Iterable[int], key: Hashable = "last", pad_with: int = 0):
        self.shape = shape
        self.key = "last"
        self.pad_with = pad_with

    def _featurize(self, system: System) -> np.ndarray:
        arraylike = np.asarray(system.featurizations[self.key])
        pads = []
        for current_size, requested_size in zip(arraylike.shape, shape):
            assert requested_size >= current_size
            pads.append([0, requested_size - current_size])
        return np.pad(arraylike, pads, mode="constant", constant_values=self.pad_with)
