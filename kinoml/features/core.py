"""
Featurizers can transform a ``kinoml.core.system.System`` object and produce
new representations of the molecular entities and their associated measurements.

All ``Featurizer`` objects inherit from ``BaseFeaturizer`` and reimplement `._featurize`
and `._supports`, if needed.
"""
from __future__ import annotations
from typing import Callable, Hashable, Iterable, Sequence
import hashlib
from multiprocessing import Pool
from functools import partial

import numpy as np

from ..core.systems import System
from ..utils import Hashabledict


class BaseFeaturizer:
    """
    Abstract Featurizer class
    """

    _SUPPORTED_TYPES = (System,)

    def __init__(self, *args, **kwargs):
        pass

    def featurize(self, systems: Iterable[System], processes=1, chunksize=None) -> object:
        """
        Given some systems (compatible with ``_SUPPORTED_TYPES``), apply
        the featurization scheme implemented in this class.

        First, ``self.supports()`` will check whether the systems are compatible
        with the featurization scheme. We assume all of them are equal, so only
        the first one will be checked. Then, the Systems are passed to
        ``self._featurize`` to handle the actual leg-work.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed
        processes : int, optional=1
            Number of processors to use. If 1, ``multiprocessing`` will not be
            used at all.
        chunksize :  int, optional=None
            See https://stackoverflow.com/a/54032744/3407590.

        Returns
        -------
        systems : list of System
            The same systems that were passed in.
            The returned Systems will have an extra entry in the ``.featurizations``
            dictionary, containing the featurized object (either a new System
            or an array-like object) under a key named after ``.name``.
        """
        self.supports(systems[0])
        features = self._featurize(systems)
        systems = self._post_featurize(systems, features)
        return systems

    def __call__(self, *args, **kwargs):
        """
        You can also call the instance directly. This forwards to
        ``.featurize()``.
        """
        return self.featurize(*args, **kwargs)

    def _featurize(self, systems: Iterable[System], processes=1, chunksize=None):
        """
        Some global properties can be optionally computed with
        ``self._featurize_options()``. This returns a dictionary that will
        be passed to ``self._featurize_one``, the method responsible of
        featurizing each System object. This part will be automatically parallelized
        if ``processes != 1``.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed
        processes : int, optional=1
            Number of processors to use. If 1, ``multiprocessing`` will not be
            used at all.
        chunksize :  int, optional=None
            See https://stackoverflow.com/a/54032744/3407590.

        Returns
        -------
        features : list of System or array-like
        """
        featurization_options = Hashabledict(self._featurize_options(systems) or {})

        if processes == 1:
            features = [self._featurize_one(s, options=featurization_options) for s in systems]
        else:
            func = partial(self._featurize_one, options=featurization_options)
            with Pool(processes=processes) as pool:
                features = pool.map(func, systems, chunksize)

        return features

    def _featurize_options(self, systems: Iterable[System]) -> dict | None:
        """
        Computes properties that depend on a collection of System objects,
        which might be needed to featurize a single system later (e.g.
        maximum length of a feature that needs to be padded).

        Parameters
        ----------
        systems : list of System
            The Systems that will be eventually featurized.

        Returns
        -------
        dict[str, object]
            Keyword arguments computed out of the list of Systems. Some
            featurizers require this dynamically computed set of options.
        """
        return None

    def _featurize_one(self, system: System, options: dict) -> object:
        """
        Implement this method to do the actual leg-work for `self.featurize()`.
        It takes a single System object and some options (see ``self._featurize_options``)
        and returns either a new System object or an array-like object.

        Parameters
        ----------
        system : System
            The System to be featurized.
        options : dict
            Keyword arguments for this featurizer, usually computed by
            ``self._featurize_options``.

        Returns
        -------
        System or array-like
        """
        raise NotImplementedError("Implement in your subclass")

    def _post_featurize(
        self, systems: Iterable[System], features: Iterable[System | np.array]
    ) -> Iterable[System]:
        """
        Run after featurizing all systems. You shouldn't need to redefine this method

        Parameters
        ----------
        systems : list of System
            The systems being featurized
        features : list of System or array
            The features returned by ``self._featurize``

        Returns
        -------
        systems
            The same systems as passed, but with ``.featurizations`` extended with
            the calculated features in two entries: the featurizer name and ``last``.
        """
        # TODO: Define self.id() to provide a unique key per class name and chosen init args
        for system, feature in zip(systems, features):
            system.featurizations[self.name] = system.featurizations["last"] = feature
        return systems

    def supports(self, *systems: System, raise_errors: bool = True) -> bool:
        """
        Check if these systems are supported by this featurizer.

        Do NOT reimplement in subclass. Check ``._supports()`` instead.

        Parameters
        ----------
        systems : list of System
            Systems to be checked (by type, contained attributes, etc)
        raise_errors: bool, optional=True
            if True, raise `ValueError` if errors were found

        Returns
        -------
        bool
            True if all systems are compatible, False otherwise

        Raises
        ------
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

    def _featurize(self, systems: Iterable[System], processes=1, chunksize=None):
        """
        Given a list of featurizers, apply them sequentially
        on the systems/arrays (e.g. featurizer A returns X, and X is
        taken by featurizer B, which returns Y).

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed
        processes : int, optional=1
            Number of processors to use. If 1, ``multiprocessing`` will not be
            used at all.
        chunksize :  int, optional=None
            See https://stackoverflow.com/a/54032744/3407590.

        Returns
        -------
        features : list of System or array-like
        """
        for featurizer in self.featurizers:
            system_or_array = featurizer._featurize(
                systems,
                processes=processes,
                chunksize=chunksize,
            )
        return system_or_array

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

    def _featurize(self, systems: Iterable[System], processes=1, chunksize=None):
        """
        Given a list of featurizers, apply them serially and concatenate
        the result (e.g. featurizer A returns X, and featurizer B returns Y;
        the output is XY).

        Parameters
        ----------
        systems_or_arrays: list of System or array-like
            The Systems (or arrays) to be featurized.
        """
        features = [
            f._featurize(systems, processes=processes, chunksize=chunksize)
            for f in self.featurizers
        ]
        return np.concatenate(features, axis=self.axis)


class BaseOneHotEncodingFeaturizer(BaseFeaturizer):
    ALPHABET = None

    def __init__(self, dictionary: dict = None):
        if dictionary is None:
            dictionary = {c: i for i, c in enumerate(self.ALPHABET)}
        self.dictionary = dictionary
        if not self.dictionary:
            raise ValueError("This featurizer requires a populated dictionary!")

    def _featurize_one(self, system: System, options: dict) -> np.ndarray:
        """
        One hot encode one system.

        Parameters
        ----------
        system : System
            The System to be featurized.
        options : dict
            Unused

        Returns
        -------
        array
        """
        sequence = self._retrieve_sequence(system)
        return self.one_hot_encode(sequence, self.dictionary)

    def _retrieve_sequence(self, system: System):
        """
        Implement in your component-specific subclass!
        """
        raise NotImplementedError

    @staticmethod
    def one_hot_encode(sequence: Iterable, dictionary: dict | Sequence) -> np.ndarray:
        """
        One-hot encode a sequence of characters, given a dictionary
        Parameters
        ----------
        sequence : Iterable
        dictionary : dict or sequuence-like
            Mapping of each character to their position in the alphabet. If
            a sequence-like is given, it will be enumerated into a dict.
        Returns
        -------
        array-like
            One-hot encoded matrix with shape ``(len(dictionary), len(sequence))``
        """
        if not isinstance(dictionary, dict):
            dictionary = {value: index for (index, value) in enumerate(dictionary)}

        ohe_matrix = np.zeros((len(dictionary), len(sequence)))
        for i, character in enumerate(sequence):
            ohe_matrix[dictionary[character], i] = 1
        return ohe_matrix


class PadFeaturizer(BaseFeaturizer):
    """
    Pads features of a given system to a desired size or length.

    This class wraps ``numpy.pad`` with ``mode=constant``, auto-calculating
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

    def _get_array(self, system_or_array: System | np.ndarray) -> np.ndarray:
        if hasattr(system_or_array, "featurizations"):
            return np.asarray(system_or_array.featurizations[self.key])
        else:
            return system_or_array

    def _featurize_options(self, systems: Iterable[System]) -> dict | None:
        """
        Compute the largest shape in the input arrays

        Parameters
        ----------
        systems : list of System

        Returns
        -------
        dict
            A dictionary containing a single key ``shape``
            with the largest shape found.
        """
        if self.shape == "auto":
            arraylikes = [self._get_array(s) for s in systems]
            return {"shape": max(a.shape for a in arraylikes)}

    def _featurize_one(self, system: System, options: dict) -> np.ndarray:
        """
        Parameters
        ----------
        system: System or array-like
            The System (or array) to be featurized.
        options: dict
            Must contain a key ``shape`` with the expected final shape
            of the systems.

        Returns
        -------
        array
        """
        shape = options.get("shape", self.shape)
        arraylike = self._get_array(system)
        pads = []
        for current_size, requested_size in zip(arraylike.shape, shape):
            assert (
                requested_size >= current_size
            ), f"{requested_size} is smaller than {current_size}!"
            pads.append([0, requested_size - current_size])
        return np.pad(arraylike, pads, mode="constant", constant_values=self.pad_with)


class HashFeaturizer(BaseFeaturizer):

    """
    Hash an attribute of the protein, such as the name or id.

    Parameters
    ----------
    getter : callable, optional
        A function or lambda that takes a System and returns
        a string to be hashed. Default value will return
        whatever ``system.featurizations["last"]`` contains,
        as a string
    normalize : bool, default=True
        Normalizes the hash to obtain a value in the unit interval
    """

    def __init__(self, getter: Callable[[System], str] = None, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.getter = getter or self._getter
        self.normalize = normalize
        self.denominator = 2 ** 256 if normalize else 1

    @staticmethod
    def _getter(system):
        return str(system.featurizations["last"])

    def _featurize_one(self, system: System, options: dict) -> np.ndarray:
        """
        Featurizes a component using the hash of the chosen attribute.

        Parameters
        ----------
        system : System
            The System to be featurized.

        Returns
        -------
        array
            Sha256'd attribute
        """
        inputdata = self.getter(system)

        h = hashlib.sha256(inputdata.encode(encoding="UTF-8"))
        return np.array(int(h.hexdigest(), base=16) / self.denominator)


class NullFeaturizer(BaseFeaturizer):
    def _featurize(self, systems: Iterable[System], processes=1, chunksize=None) -> object:
        return systems


class CallableFeaturizer(BaseFeaturizer):
    """
    Apply an arbitrary callable to a System.

    Parameters
    ----------
    func : callable
        Must take a System and return a System or array.
    """

    def __init__(self, func: Callable[[System], System | np.array]):
        self.callable = func

    def _featurize_one(self, system: System | np.ndarray, options: dict) -> np.ndarray:
        """
        Parameters
        ----------
        system: System or array-like
            The System (or array) to be featurized.
        options : dict
            Unused

        Returns
        -------
        array-like
        """
        return self.callable(system)


class ClearFeaturizations(BaseFeaturizer):
    """
    Remove keys from the ``.featurizations`` dictionary in each
    ``System`` object. By default, it will remove all keys
    that are not ``last``.

    Parameters
    ----------
    keys : tuple of str, optional=("last",)
        Which keys to keep or remove, depending on ``style``.
    style : str, optional="keep"
        Whether to ``keep`` or ``remove`` the entries passed as ``keys``.
    """

    def __init__(self, keys=("last",), style="keep"):
        assert style in ("keep", "remove"), "`style` must be `keep` or `remove`"
        self.keys = keys
        self.style = style

    def _featurize_one(self, system: System, options: dict) -> System:
        if self.style == "keep":
            to_remove = [k for k in system.featurizations.keys() if k not in self.keys]
        else:
            to_remove = self.keys

        for key in to_remove:
            system.featurizations.pop(key, None)

        return system

    def _post_featurize(
        self, systems: Iterable[System], features: Iterable[System | np.array]
    ) -> Iterable[System]:
        """
        Bypass the automated population of the ``.featurizations`` dict
        in each System
        """
        return systems