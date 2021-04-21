"""
Featurizers can transform a ``kinoml.core.system.System`` object and produce
new representations of the molecular entities and their associated measurements.

All ``Featurizer`` objects inherit from ``BaseFeaturizer`` and reimplement `._featurize`
and `._supports`, if needed.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Hashable, Iterable, Union
from functools import lru_cache
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

    def featurize(self, systems: Iterable[System]) -> object:
        """
        Given some systems (compatible with ``_SUPPORTED_TYPES``), apply
        the featurization scheme implemented in this class.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed

        Returns
        -------
        system: System
            The same system that was passed in, unless ``inplace`` is False.
            The returned System will have an extra entry in the ``.featurizations``
            dictionary, containing the featurized object (either a new System
            or an array-like object) undera key named after ``.name``.
        """
        self.supports(systems[0])
        features = self._featurize(tuple(systems))
        # TODO: Define self.id() to provide a unique key per class name and chosen init args
        for system, feature in zip(systems, features):
            system.featurizations[self.name] = feature
        return systems

    def __call__(self, *args, **kwargs):
        """
        You can also call the instance directly. This forwards to
        ``.featurize()``.
        """
        return self.featurize(*args, **kwargs)

    def _featurize(self, systems: Iterable[System]) -> object:
        """
        Implement this method to do the actual work for `self.featurize()`.

        Parameters
        ----------
        systems: list of System
            The Systems to be featurized.
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
    Given a list of featurizers, apply them sequentially
    on the systems (e.g. featurizer A returns X, and X is
    taken by featurizer B, which returns Y).

    Parameters
    ----------
    featurizers: list of BaseFeaturizer
        Featurizers to stack. They must be compatible with
        each other!
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer]):
        self.featurizers = featurizers

    def _featurize(self, systems_or_arrays: Iterable[Union[System, np.ndarray]]):
        """
        Given a list of featurizers, apply them sequentially
        on the systems/arrays (e.g. featurizer A returns X, and X is
        taken by featurizer B, which returns Y).

        Parameters
        ----------
        systems_or_arrays: list of System or array-like
            The Systems (or arrays) to be featurized.
        """
        for featurizer in self.featurizers:
            systems_or_arrays = featurizer._featurize(systems_or_arrays)
        return systems_or_arrays

    def supports(self, *systems: System, raise_errors: bool = False) -> bool:
        """
        Check if these systems are supported by all featurizers.

        Parameters
        ----------
        systems : list of System
            systems to be checked (by type, contained attributes, etc)
        raise_errors : bool, optional=False
            If True, raise ``ValueError``

        Returns
        -------
        bool:
            True if all systems are compatible with all featurizers, False otherwise

        Raises
        ------
        ``ValueError`` if ``f.supports()`` fails and ``raise_errors`` is ``True``.
        """
        return all(
            f.supports(s, raise_errors=raise_errors) for f in self.featurizers for s in systems
        )

    @property
    def name(self):
        return f"{self.__class__.__name__}([{', '.join([f.name for f in self.featurizers])}])"


class Concatenated(Pipeline):
    """
    Given a list of featurizers, apply them serially and concatenate
    the result (e.g. featurizer A returns X, and featurizer B returns Y;
    the output is XY).

    Parameters
    ----------
    featurizers : list of BaseFeaturizer
        These should take a System or array, but return only arrays
        so the can be concatenated.
    axis : int, optional=0
        On which axis to concatenate
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer], axis=0):
        self.featurizers = featurizers
        self.axis = axis

    def _featurize(self, systems_or_arrays: Iterable[Union[System, np.ndarray]]):
        """
        Given a list of featurizers, apply them serially and concatenate
        the result (e.g. featurizer A returns X, and featurizer B returns Y;
        the output is XY).

        Parameters
        ----------
        systems_or_arrays: list of System or array-like
            The Systems (or arrays) to be featurized.
        """
        features = [f._featurize(systems_or_arrays) for f in self.featurizers]
        return np.concatenate(features, axis=self.axis)


class BaseOneHotEncodingFeaturizer(BaseFeaturizer):
    ALPHABET = None

    def __init__(self, dictionary: dict = None):
        if dictionary is None:
            dictionary = {c: i for i, c in enumerate(self.ALPHABET)}
        self.dictionary = dictionary
        if not self.dictionary:
            raise ValueError("This featurizer requires a populated dictionary!")

    def _featurize(self, systems: Iterable[System]):
        """
        Parameters
        ----------
        systems: list of System
            The Systems to be featurized.
        """
        matrices = []
        for system in systems:
            sequence = self._retrieve_sequence(system)
            matrices.append(self.one_hot_encode(sequence, self.dictionary))
        return matrices

    def _retrieve_sequence(self, system: System):
        """
        Implement in your component-specific subclass!
        """
        raise NotImplementedError

    @staticmethod
    def one_hot_encode(sequence: str, dictionary: dict) -> np.ndarray:
        """
        One-hot encode a sequence of characters, given a dictionary

        Parameters
        ----------
        sequence : str
        dictionary : dict
            Mapping of each character to their position in the alphabet

        Returns
        -------
        array-like
            One-hot encoded matrix with shape ``(len(dictionary), len(sequence))``
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

    Parameters
    ----------
    shape : tuple of int, or "auto"
        The desired size of the transformed features. If "auto", shape
        will be estimated from the Dataset passed at runtime so it matches
        the largest observed.
    key : hashable
        element to retrieve from ``System.featurizations``
    pad_with : int
        value to fill the array-like features with
    """

    def __init__(self, shape: Iterable[int] = "auto", key: Hashable = "last", pad_with: int = 0):
        self.shape = shape
        self.key = key
        self.pad_with = pad_with

    def _featurize(self, systems_or_arrays: Iterable[Union[System, np.ndarray]]) -> np.ndarray:
        """
        Parameters
        ----------
        systems_or_arrays: list of System or array-like
            The Systems (or arrays) to be featurized.
        """
        if hasattr(systems_or_arrays[0], "featurizations"):
            arraylikes = [np.asarray(sa.featurizations[self.key]) for sa in systems_or_arrays]
        else:
            arraylikes = systems_or_arrays

        shape = max(a.shape for a in arraylikes) if self.shape == "auto" else self.shape

        padded = []
        for arraylike in arraylikes:
            pads = []
            for current_size, requested_size in zip(arraylike.shape, shape):
                assert (
                    requested_size >= current_size
                ), f"{requested_size} is smaller than {current_size}!"
                pads.append([0, requested_size - current_size])
            padded.append(np.pad(arraylike, pads, mode="constant", constant_values=self.pad_with))
        return padded


class HashFeaturizer(BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.

    Parameters
    ----------
    attribute : str or tuple
        Attribute(s) in the target object that will be hashed
    normalize : bool, default=True
        Normalizes the hash to obtain a value in the unit interval
    """

    def __init__(self, attributes, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = attributes
        self.normalize = normalize

    def _featurize(self, systems: Iterable[System]):
        """
        Featurizes a component using the hash of the chosen attribute.

        Parameters
        ----------
        systems : list of System
            The Systems to be featurized.

        Returns
        -------
        list of array
            Sha256'd attributes
        """
        results = []
        denominator = 2 ** 256 if self.normalize else 1
        for system in systems:
            inputdata = system
            for attr in self.attributes:
                inputdata = getattr(inputdata, attr)

            h = hashlib.sha256(inputdata.encode(encoding="UTF-8"))
            result = np.reshape(int(h.hexdigest(), base=16) / denominator, (1,))
            results.append(result)
        return results


class NullFeaturizer(BaseFeaturizer):
    def featurize(self, systems: Iterable[System]) -> Iterable[System]:
        return systems


class ScaleFeaturizer(BaseFeaturizer):
    """
    WIP
    """

    def __init__(self, key: Hashable = "last", **kwargs):
        self.key = key
        self.sklearn_options = kwargs

    def _featurize(self, systems_or_arrays: Iterable[Union[System, np.ndarray]]) -> np.ndarray:
        """
        Parameters
        ----------
        systems_or_arrays: list of System or array-like
            The System to be featurized.

        Returns
        -------
        list of array-like
        """
        from sklearn.preprocessing import scale

        if hasattr(systems_or_arrays[0], "featurizations"):
            arraylikes = [np.asarray(sa.featurizations[self.key]) for sa in systems_or_arrays]
        else:
            arraylikes = systems_or_arrays
        return [scale(arr, **self.sklearn_options) for arr in arraylikes]
