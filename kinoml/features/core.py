"""
Featurizers can transform a ``kinoml.core.system.System`` object and produce
new representations of the molecular entities and their associated measurements.
"""
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Sequence, Union
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from tqdm.auto import tqdm

from ..core.systems import System


class BaseFeaturizer:
    """
    Abstract Featurizer class.
    """

    _SUPPORTED_TYPES = (System,)

    def featurize(
            self,
            systems: Iterable[System],
            keep=True,
    ) -> Iterable[System]:
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
            This is the collection of System objects that will be transformed.
        keep : bool, optional=True
            Whether to store the current featurizer in the ``system.featurizations``
            dictionary with its own key (``self.name``), in addition to ``last``.

        Returns
        -------
        systems : list of System
            The same systems that were passed in.
            The returned Systems will have an extra entry in the ``.featurizations``
            dictionary, containing the featurized object (either a new System
            or an array-like object) under a key named after ``.name``.
        """
        self.supports(systems[0])
        self._pre_featurize(systems)
        features = self._featurize(systems)
        systems = self._post_featurize(systems, features, keep=keep)
        return systems

    def __call__(self, *args, **kwargs):
        """
        You can also call the instance directly. This forwards to
        ``.featurize()``.
        """
        return self.featurize(*args, **kwargs)

    def _pre_featurize(self, systems: Iterable[System]) -> None:
        """
        Run before featurizing all systems. Redefine this method if needed.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed.
        """
        return

    def _featurize(self, systems: Iterable[System]) -> Iterable[object]:
        """
        Featurize all system objects in a serial fashion as defined in ``._featurize_one()``.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed.

        Returns
        -------
        features : list of System or array-like
        """
        features = [self._featurize_one(s) for s in tqdm(systems, desc=self.name)]
        return features

    def _featurize_one(self, system: System) -> object:
        """
        Implement this method to do the actual leg-work for `self.featurize()`.
        It takes a single System object and returns either a new System object
        or an array-like object.

        Parameters
        ----------
        system : System
            The System to be featurized.

        Returns
        -------
        System or array-like
        """
        raise NotImplementedError("Implement in your subclass")

    def _post_featurize(
        self, systems: Iterable[System], features: Iterable[System | np.array], keep: bool = True
    ) -> Iterable[System]:
        """
        Run after featurizing all systems. You shouldn't need to redefine this method

        Parameters
        ----------
        systems : list of System
            The systems being featurized
        features : list of System or array
            The features returned by ``self._featurize``
        keep : bool, optional=True
            Whether to store the current featurizer in the ``system.featurizations``
            dictionary with its own key (``self.name``), in addition to ``last``.

        Returns
        -------
        systems
            The same systems as passed, but with ``.featurizations`` extended with
            the calculated features in two entries: the featurizer name and ``last``.
        """
        # TODO: Define self.id() to provide a unique key per class name and chosen init args
        for system, feature in zip(systems, features):
            system.featurizations["last"] = feature
            if keep:
                system.featurizations[self.name] = feature
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

        Parameters
        ----------
        system: System
            The system that will be checked

        Returns
        -------
        True if compatible, False otherwise
        """
        return isinstance(system, self._SUPPORTED_TYPES)

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.name}>"


class ParallelBaseFeaturizer(BaseFeaturizer):
    """
    Abstract Featurizer class with support for multiprocessing.

    Parameters
    ----------
    use_multiprocessing : bool, default=True
        If multiprocessing to use.
    n_processes : int or None, default=None
        How many processes to use in case of multiprocessing.
        Defaults to number of available CPUs.
    chunksize :  int, optional=None
        See https://stackoverflow.com/a/54032744/3407590.
    dask_client : dask.distributed.Client or None, default=None
        A dask client to manage multiprocessing. Will ignore `use_multiprocessing`
        `chunksize` and `n_processes` attributes.
    """

    _SUPPORTED_TYPES = (System,)

    # TODO: environment variables for multiprocessing

    def __init__(
            self,
            use_multiprocessing: bool = True,
            n_processes: Union[int, None] = None,
            chunksize: Union[int, None] = None,
            dask_client=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_multiprocessing = use_multiprocessing
        self.n_processes = n_processes
        self.chunksize = chunksize
        self.dask_client = dask_client

    def __getstate__(self):
        """Only preserve object fields that are serializable"""

        def is_serializable(value):
            import pickle
            try:
                pickle.dumps(value)
                return True
            except AttributeError as e:
                return False

        return {name: value for name, value in self.__dict__.items() if is_serializable(value)}

    def __setstate__(self, state):
        """Only preserve object fields that are serializable."""
        for name, value in state.items():
            setattr(self, name, value)

    def _featurize(self, systems: Iterable[System]) -> Iterable[object]:
        """
        Featurize all system objects in a parallel fashion as defined in ``._featurize_one()``.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed.

        Returns
        -------
        features : list of System or array-like
        """
        # check for multiprocessing options and featurize
        if self.dask_client is not None:
            # check if dask_client is a Client from dask.distributed
            if not hasattr(self.dask_client, "map"):
                from dask.distributed import Client
                if not isinstance(self.dask_client, Client):
                    raise ValueError(
                        "The dask_client attribute appears not to be a Client from dask.distributed."
                    )
            # featurize in parallel with dask
            func = partial(self._featurize_one)
            futures = self.dask_client.map(func, systems)
            features = self.dask_client.gather(futures)
        else:
            # determine the number of processes to spawn
            if self.use_multiprocessing:
                if not self.n_processes:
                    self.n_processes = cpu_count()
            else:
                self.n_processes = 1
            if self.n_processes == 1:
                # featurize in a serial fashion
                features = [
                    self._featurize_one(s)
                    for s in tqdm(systems, desc=self.name)
                ]
            else:
                # featurize in a parallel fashion
                func = partial(self._featurize_one)
                with Pool(processes=self.n_processes) as pool:
                    features = pool.map(func, systems, self.chunksize)

        return features


class Pipeline(BaseFeaturizer):

    """
    Given a list of featurizers, apply them sequentially
    on the systems (e.g. featurizer A returns X, and X is
    taken by featurizer B, which returns Y).

    Parameters
    ----------
    featurizers: iterable of BaseFeaturizer
        Featurizers to stack. They must be compatible with
        each other!

    Note
    ----
    While ``Pipeline`` is a subclass of ``BaseFeaturizer``,
    it should be considered a special case of such. It indeed
    shares the same API but the implementation details of
    ``._featurize()`` are slightly different. It acts as a
    wrapper around individual ``Featurizer`` objects.
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer], shortname=None, **kwargs):
        super().__init__(**kwargs)
        self.featurizers = featurizers
        self._shortname = shortname

    def _featurize(self, systems: Iterable[System], keep: bool = True) -> Iterable[object]:
        """
        Given a list of featurizers, apply them sequentially
        on the systems (e.g. featurizer A returns X, and X is
        taken by featurizer B, which returns Y) and store the
        features in the systems.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed
        keep : bool, optional=True
            Whether to store the current featurizer in the ``system.featurizations``
            dictionary with its own key (``self.name``), in addition to ``last``.

        Returns
        -------
        features : list of System or array-like
        """
        for featurizer in self.featurizers:
            systems = featurizer.featurize(systems, keep=keep)

        return [s.featurizations["last"] for s in systems]

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
        if self._shortname:
            return (
                f"{self.__class__.__name__}(name='{self.shortname}', "
                f"[{', '.join([f.name for f in self.featurizers])}])"
            )
        return f"{self.__class__.__name__}([{', '.join([f.name for f in self.featurizers])}])"

    @property
    def shortname(self):
        if self._shortname is not None:
            return self._shortname
        return self.__class__.__name__


class Concatenated(Pipeline):
    """
    Given a list of featurizers, apply them serially and concatenate
    the result (e.g. featurizer A returns X, and featurizer B returns Y;
    the output is XY).

    Parameters
    ----------
    featurizers : list of BaseFeaturizer
        These should take a System or array, but return only arrays
        so they can be concatenated. Note that the arrays must
        have the same number of dimensions. If that is not the case,
        you will need to reshape one of them using ``CallableFeaturizer``
        and a lambda function that relies on ``np.reshape`` or similar.
    axis : int, optional=1
        On which axis to concatenate. By default, it will concatenate
        on axis ``1``, which means that the features in each pipeline
        will be concatenated.

    Notes
    -----
    This Featurizer maybe removed in the future, since it can be replaced
    by `TupleOfArrays`.
    """

    def __init__(self, featurizers: Iterable[BaseFeaturizer], axis: int = 1, **kwargs):
        super().__init__(featurizers, **kwargs)
        self.axis = axis

    def _featurize(self, systems: Iterable[System], keep=True) -> np.ndarray:
        """
        Given a list of featurizers, apply them serially and concatenate
        the result (e.g. featurizer A returns X, and featurizer B returns Y;
        the output is XY).

        Parameters
        ----------
        systems: list of System or array-like
            The Systems (or arrays) to be featurized.
        keep : bool, optional=True
            Whether to store the current featurizer in the ``system.featurizations``
            dictionary with its own key (``self.name``), in addition to ``last``.

        Returns
        -------
        np.ndarray
            Concatenated arrays along specified ``axis``.
        """
        # Concatenation expects a list of features to be concatenated
        # Each list of features comes from a different pipeline
        # Within a list, we find one feature (array) for each system
        # We need to end up with a single array!
        list_of_features = []
        for featurizer in self.featurizers:
            systems = featurizer.featurize(systems, keep=keep)
            features = [s.featurizations["last"] for s in systems]
            list_of_features.append(features)

        return np.concatenate(list_of_features, axis=self.axis)


class TupleOfArrays(Pipeline):
    """
    Given a list of featurizers, apply them serially and return
    the result directly as a flattened tuple of the arrays, for
    each system. E.g; given one system, featurizer A returns X,
    and featurizer B returns Y, Z; the output is a tuple of X, Y, Z).

    The final result will be tuple of tuples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _featurize(self, systems: Iterable[System], keep: bool = True) -> np.ndarray:
        """
        Given a list of featurizers, apply them serially and build a
        flat tuple out of the results.

        Parameters
        ----------
        systems: list of System or array-like
            The Systems (or arrays) to be featurized.
        keep : bool, optional=True
            Whether to store the current featurizer in the ``system.featurizations``
            dictionary with its own key (``self.name``), in addition to ``last``.

        Returns
        -------
        : tuple of (of tuples) arraylike
            If the last featurizer is returning a single array,
            the shape of the object will be (N_systems,). If
            the last featurizer returns more than one array,
            it will be (N_systems, M_returned_objects).
        """
        list_of_systems = []
        for _ in systems:
            list_of_systems.append([])

        features_per_system = 0
        for featurizer in self.featurizers:
            # Run a pipeline
            systems = featurizer.featurize(systems, keep=keep)
            # Get the current "last" before the next pipeline runs
            # We need to store it in list_of_systems
            # list of systems will have shape (N_systems, n_featurizers)
            for i, system in enumerate(systems):
                features = system.featurizations["last"]
                if isinstance(features, dict):
                    if i == 0:
                        features_per_system += len(features)
                    for array in features.values():
                        assert isinstance(
                            array, np.ndarray
                        ), f"Array {array} is not a ndarray type!"
                        list_of_systems[i].append(array)
                elif isinstance(features, (tuple, list)):
                    if i == 0:
                        features_per_system += len(features)
                    for array in features:
                        assert isinstance(
                            array, np.ndarray
                        ), f"Array {array} is not a ndarray type!"
                        list_of_systems[i].append(array)
                elif isinstance(features, np.ndarray):
                    if i == 0:
                        features_per_system += 1
                    # no extra dimension needed when
                    # the returned object is a single array
                    list_of_systems[i].append(features)
                else:
                    raise ValueError(
                        f"Obtained features ({features}) is not recognized. It must "
                        "be ndarray, or a tuple/list/dict of ndarray"
                    )

        assert len(list_of_systems) == len(
            systems
        ), f"Number of feature tuples ({len(list_of_systems)}) do not match systems ({len(systems)}!"
        assert (
            len(list_of_systems[0]) == features_per_system
        ), f"Number of features per system ({len(list_of_systems[0])}) do not match number of expected ({features_per_system})!"
        return list_of_systems


class BaseOneHotEncodingFeaturizer(ParallelBaseFeaturizer):
    ALPHABET = None

    def __init__(self, dictionary: dict = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is None:
            dictionary = {c: i for i, c in enumerate(self.ALPHABET)}
        self.dictionary = dictionary
        if not self.dictionary:
            raise ValueError("This featurizer requires a populated dictionary!")

    def _featurize_one(self, system: System) -> np.ndarray:
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
        One-hot encode a sequence of characters, given a dictionary.

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


class PadFeaturizer(ParallelBaseFeaturizer):
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

    def __init__(self, shape: Iterable[int] = "auto", key: Hashable = "last", pad_with: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.key = key
        self.pad_with = pad_with

    def _get_array(self, system_or_array: System | np.ndarray) -> np.ndarray:
        if hasattr(system_or_array, "featurizations"):
            return np.asarray(system_or_array.featurizations[self.key])
        else:
            return system_or_array

    def _pre_featurize(self, systems) -> None:
        """
        Compute the largest shape in the input arrays and store in shape attribute.

        Parameters
        ----------
        systems : list of System
        """
        if self.shape == "auto":
            arraylikes = [self._get_array(s) for s in systems]
            self.shape = max(a.shape for a in arraylikes)

    def _featurize_one(self, system: System) -> np.ndarray:
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
        arraylike = self._get_array(system)
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
    ----------
    getter : callable, optional
        A function or lambda that takes a System and returns
        a string to be hashed. Default value will return
        whatever ``system.featurizations["last"]`` contains,
        as a string
    normalize : bool, default=True
        Normalizes the hash to obtain a value in the unit interval
    """

    def __init__(self, getter: Callable[[System], str] = None, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.getter = getter or self._getter
        self.normalize = normalize
        self.denominator = 2 ** 256 if normalize else 1

    @staticmethod
    def _getter(system):
        return str(system.featurizations["last"])

    def _featurize_one(self, system: System) -> np.ndarray:
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
        return np.reshape(np.array(int(h.hexdigest(), base=16) / self.denominator), -1)


class NullFeaturizer(ParallelBaseFeaturizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _featurize(self, systems: Iterable[System], keep: bool = None) -> object:
        return systems


class CallableFeaturizer(BaseFeaturizer):
    """
    Apply an arbitrary callable to a System.

    Parameters
    ----------
    func : callable or str or None
        Must take a System and return a System or array. If
        ``str`` it will be ``eval``'d into a callable. If None,
        the default callable will return ``system.featurizations["last"]``
        for each system.
    """

    def __init__(self, func: Callable[[System], System | np.array] | str = None, **kwargs):
        super().__init__(**kwargs)
        if func is None:
            func = self._default_func
        elif isinstance(func, str):
            func = eval(func)  # pylint: disable=eval-used
        self.callable = func

    @staticmethod
    def _default_func(system):
        return system.featurizations["last"]

    def _featurize_one(self, system: System | np.ndarray) -> np.ndarray:
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

    def __init__(self, keys=("last",), style="keep", **kwargs):
        super().__init__(**kwargs)
        assert style in ("keep", "remove"), "`style` must be `keep` or `remove`"
        self.keys = keys
        self.style = style

    def _featurize_one(self, system: System) -> System:
        if self.style == "keep":
            to_remove = [k for k in system.featurizations.keys() if k not in self.keys]
        else:
            to_remove = self.keys

        for key in to_remove:
            system.featurizations.pop(key, None)

        return system

    def _post_featurize(
        self, systems: Iterable[System], features: Iterable[System | np.array], keep: bool = True
    ) -> Iterable[System]:
        """
        Bypass the automated population of the ``.featurizations`` dict
        in each System
        """
        return systems
