"""
Featurizers can transform a ``kinoml.core.system.System`` object and produce
new representations of the molecular entities and their associated measurements.

All ``Featurizer`` objects inherit from ``BaseFeaturizer`` and reimplement `._featurize`
and `._supports`, if needed.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Hashable, Iterable, Union
import hashlib

import numpy as np

from ..core.systems import System


class BaseFeaturizer:
    """
    Abstract Featurizer class
    """

    _SUPPORTED_TYPES = (System,)

    def __init__(self, *args, **kwargs):
        pass

    def featurize(self, system, inplace: bool = True) -> object:
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
        features = self._featurize(system)
        # TODO: Define self.id() to provide a unique key per class name and chosen init args
        system.featurizations[self.name] = features
        return system

    def __call__(self, *args, **kwargs):
        return self.featurize(*args, **kwargs)

    def _featurize(self, system: System) -> object:
        """
        Implement this method to do the actual work for `self.featurize()`.
        """
        raise NotImplementedError("Implement in your subclass")

    def supports(self, *systems: System, raise_errors: bool = True) -> bool:
        """
        Check if these systems are supported by this featurizer.

        Do NOT reimplement in subclass. Check ``._supports()`` instead.

        Parameters:
            systems: systems to be checked (by type, contained attributes, etc)
            raise_errors: if True, raise `ValueError`.

        Returns:
            True if all systems are compatible, False otherwise

        Raises:
            ``ValueError`` if ``._supports()`` fails and ``raise_errors`` is `True`.
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
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.name}>"


class Pipeline(BaseFeaturizer):

    """
    Featurizer-compatible class that provides a way to stack
    featurizers together

    Parameters:
        featurizers: featurizers to stack. They must be compatible!
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer]):
        self.featurizers = featurizers

    def _featurize(self, system_or_array):
        for featurizer in self.featurizers:
            system_or_array = featurizer._featurize(system_or_array)
        return system_or_array

    def supports(self, *systems: System, raise_errors: bool = False) -> bool:
        """
        Check if these systems are supported by all featurizers.

        Parameters:
            systems: systems to be checked (by type, contained attributes, etc)
            raise_errors: if True, raise `ValueError`.

        Returns:
            True if all systems are compatible with all featurizers, False otherwise

        Raises:
            ``ValueError`` if ``f.supports()`` fails and ``raise_errors`` is `True`.
        """
        return all(
            f.supports(s, raise_errors=raise_errors) for f in self.featurizers for s in systems
        )

    @property
    def name(self):
        return f"{self.__class__.__name__}([{', '.join([f.name for f in self.featurizers])}])"


class Concatenated(Pipeline):
    def __init__(self, featurizers: Iterable[BaseFeaturizer], axis=0):
        self.featurizers = featurizers
        self.axis = axis

    def _featurize(self, system_or_array):
        features = [f._featurize(system_or_array) for f in self.featurizers]
        return np.concatenate(features, axis=self.axis)


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

    This class wraps ``numpy.pad`` with `mode=constant`, auto-calculating
    the needed additions to match the requested shape.

    Parameters:
        shape: the desired size of the transformed features
        key: element to retrieve from `System.featurizations`
        pad_with: value to fill the array-like features with
    """

    def __init__(self, shape: Iterable[int], key: Hashable = "last", pad_with: int = 0):
        self.shape = shape
        self.key = key
        self.pad_with = pad_with

    def _featurize(self, system_or_array: Union[System, np.ndarray]) -> np.ndarray:
        if hasattr(system_or_array, "featurizations"):
            arraylike = np.asarray(system_or_array.featurizations[self.key])
        else:
            arraylike = system_or_array
        pads = []
        for current_size, requested_size in zip(arraylike.shape, self.shape):
            assert (
                requested_size >= current_size
            ), f"{requested_size} is smaller than {current_size}!"
            pads.append([0, requested_size - current_size])
        return np.pad(arraylike, pads, mode="constant", constant_values=self.pad_with)


class HashFeaturizer(BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.

    Parameters
    ==========
    attribute : str or tuple
        Attribute(s) in the target object that will be hashed
    normalize : bool, default=True
        Normalizes the hash to obtain a value in the unit interval
    """

    def __init__(self, attributes, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = attributes
        self.normalize = normalize

    def _featurize(self, system):
        """
        Featurizes a component using the hash of the chosen attribute.

        Returns:
            Sha256'd attribute
        """
        inputdata = system
        for attr in self.attributes:
            inputdata = getattr(inputdata, attr)

        h = hashlib.sha256(inputdata.encode(encoding="UTF-8"))
        if self.normalize:
            return np.reshape(int(h.hexdigest(), base=16) / 2 ** 256, (1,))
        return np.reshape(int(h.hexdigest(), base=16), (1,))


class NullFeaturizer(BaseFeaturizer):
    def featurize(self, system, inplace: bool = True) -> object:
        return system


class ScaleFeaturizer(BaseFeaturizer):
    def __init__(self, key: Hashable = "last", **kwargs):
        self.key = key
        self.sklearn_options = kwargs

    def _featurize(self, system_or_array: Union[System, np.ndarray]) -> np.ndarray:
        from sklearn.preprocessing import scale

        if hasattr(system_or_array, "featurizations"):
            arraylike = np.asarray(system_or_array.featurizations[self.key])
        else:
            arraylike = system_or_array

        return scale(arraylike, **self.sklearn_options)
