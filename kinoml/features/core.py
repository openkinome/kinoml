"""
Featurizers can transform a ``kinoml.core.system.System`` object and produce
new representations of the molecular entities and their associated measurements.
"""
from __future__ import annotations

from functools import partial
import hashlib
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Hashable, Iterable, Sequence, Union, Tuple, List

import numpy as np
from tqdm.auto import tqdm

from ..core.sequences import AminoAcidSequence
from ..core.systems import System, ProteinSystem, ProteinLigandComplex


class BaseFeaturizer:
    """
    Abstract Featurizer class.
    """

    _SUPPORTED_TYPES = (System,)

    def featurize(
            self,
            systems: List[System],
            keep=True,
    ) -> List[System]:
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

    def _pre_featurize(self, systems: List[System]) -> None:
        """
        Run before featurizing all systems. Redefine this method if needed.

        Parameters
        ----------
        systems : list of System
            This is the collection of System objects that will be transformed.
        """
        return

    def _featurize(self, systems: List[System]) -> List[object]:
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
        self, systems: List[System], features: List[System | np.array], keep: bool = True
    ) -> List[System]:
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

    def _featurize(self, systems: List[System]) -> List[object]:
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

    def __init__(self, featurizers: List[BaseFeaturizer], shortname=None, **kwargs):
        super().__init__(**kwargs)
        self.featurizers = featurizers
        self._shortname = shortname

    def _featurize(self, systems: List[System], keep: bool = True) -> List[object]:
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

    def __init__(self, featurizers: List[BaseFeaturizer], axis: int = 1, **kwargs):
        super().__init__(featurizers, **kwargs)
        self.axis = axis

    def _featurize(self, systems: List[System], keep=True) -> np.ndarray:
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

    def _featurize(self, systems: List[System], keep: bool = True) -> List:
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


class OEBaseModelingFeaturizer(ParallelBaseFeaturizer):
    """
    This abstract class defines several methods that use functionality from the OpenEye toolkit
    for molecular modeling. Featurizers that subclass `OEBaseModelingFeaturizer` need to implement
    at least the `_featurize_one` method.

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    cache_dir: str, Path or None, default=None
        Path to directory used for saving intermediate files. If None, default location
        provided by `appdirs.user_cache_dir()` will be used.
    output_dir: str, Path or None, default=None
        Path to directory used for saving output files. If None, output structures will not be
        saved.
    """
    from openeye import oechem

    def __init__(
            self,
            loop_db: Union[str, None] = None,
            cache_dir: Union[str, Path, None] = None,
            output_dir: Union[str, Path, None] = None,
            **kwargs,
    ):
        from appdirs import user_cache_dir

        super().__init__(**kwargs)
        self.loop_db = loop_db
        self.cache_dir = Path(user_cache_dir())
        self.output_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if output_dir:
            self.output_dir = Path(output_dir).expanduser().resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _interpret_system(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> dict:
        """
        Interpret the attributes of the given system components and store them in a dictionary.

        Parameters
        ----------
        system: ProteinSystem or ProteinLigandComplex
            The system to interpret.

        Returns
        -------
        : dict
            A dictionary containing the content of the system components.
        """
        from ..databases.pdb import download_pdb_structure

        system_dict = {
            "protein_name": None,
            "protein_pdb_id": None,
            "protein_path": None,
            "protein_sequence": None,
            "protein_uniprot_id": None,
            "protein_chain_id": None,
            "protein_alternate_location": None,
            "protein_expo_id": None,
            "ligand_name": None,
            "docking_template_pdb_id": None,
            "docking_template_path": None,
            "docking_template_expo_id": None,
            "docking_template_chain_id": None,
            "docking_template_alternate_location": None,
        }

        logging.debug("Interpreting protein component ...")
        if hasattr(system.protein, "name"):
            system_dict["protein_name"] = system.protein.name

        if hasattr(system.protein, "pdb_id"):
            system_dict["protein_path"] = download_pdb_structure(
                system.protein.pdb_id, self.cache_dir
            )
            if not system_dict["protein_path"]:
                raise ValueError(
                    f"Could not download structure for PDB entry {system.protein.pdb_id}."
                )
        elif hasattr(system.protein, "path"):
            system_dict["protein_path"] = Path(system.protein.path).expanduser().resolve()
        else:
            raise AttributeError(
                f"The {self.__class__.__name__} requires systems with protein components having a"
                f" `pdb_id` or `path` attribute."
            )

        if not hasattr(system.protein, "sequence"):
            if hasattr(system.protein, "uniprot_id"):
                logging.debug(
                    f"Retrieving amino acid sequence details for UniProt entry "
                    f"{system.protein.uniprot_id} ..."
                )
                system_dict["protein_sequence"] = AminoAcidSequence.from_uniprot(
                    system.protein.uniprot_id
                )
                system_dict["protein_uniprot_id"] = system.protein.uniprot_id
        else:
            if not isinstance(system.protein.sequence, AminoAcidSequence):
                raise AttributeError(
                    f"The {self.__class__.__name__} only accepts systems with protein components whose"
                    f" `sequence` attribute is an instance of `core.sequences.AminoAcidSequence`."
                )
            else:
                system_dict["protein_sequence"] = system.protein.sequence

        if hasattr(system.protein, "chain_id"):
            system_dict["protein_chain_id"] = system.protein.chain_id

        if hasattr(system.protein, "alternate_location"):
            system_dict["protein_alternate_location"] = system.protein.alternate_location

        if hasattr(system.protein, "expo_id"):
            system_dict["protein_expo_id"] = system.protein.expo_id

        if hasattr(system, "ligand"):
            logging.debug("Interpreting ligand component ...")
            if hasattr(system.ligand, "name"):
                system_dict["ligand_name"] = system.ligand.name

            if hasattr(system.ligand, "docking_template_pdb_id"):
                system_dict["docking_template_pdb_id"] = system.ligand.docking_template_pdb_id
                system_dict["docking_template_path"] = download_pdb_structure(
                    system.ligand.docking_template_pdb_id, self.cache_dir
                )
                if not system_dict["docking_template_path"]:
                    raise ValueError(
                        f"Could not download structure for PDB entry "
                        f"{system.ligand.docking_template_pdb_id}."
                    )

            elif hasattr(system.ligand, "docking_template_path"):
                system_dict["docking_template_path"] = system.ligand.docking_template_path

            if hasattr(system.ligand, "docking_template_expo_id"):
                system_dict["docking_template_expo_id"] = system.ligand.docking_template_expo_id

            if hasattr(system.ligand, "docking_template_chain_id"):
                system_dict["docking_template_chain_id"] = system.ligand.docking_template_chain_id

            if hasattr(system.ligand, "docking_template_alternate_location"):
                system_dict["docking_template_alternate_location"] = \
                    system.ligand.docking_template_alternate_location

        return system_dict

    def _get_design_unit(
            self,
            structure: oechem.OEMolBase,
            chain_id: Union[str, None],
            alternate_location: Union[str, None],
            has_ligand: bool,
            ligand_name: Union[str, None],
            model_loops_and_caps: bool,
    ) -> Union[oechem.OEDesignUnit, None]:
        """
        Get an OpenEye design unit based on the given input.

        Parameters
        ----------
        structure: oechem.OEMolBase
            An OpenEye molecule holding the protein structure to prepare.
        chain_id: str or None
            The chain ID of interest.
        alternate_location: str or None
            The alternate location of interest.
        has_ligand: bool
            If design unit generation should consider ligands. If True, design units will be only
            generated for protein ligand complexes. If False, design units will not consider
            co-crystallized ligands.
        ligand_name: str or None
            The ligand expo ID bound to the protein of interest. Design units will be filtered to
            contain the respective ligand.
        model_loops_and_caps: bool
            If loops and caps should be modeled.

        Returns
        -------
        design_unit: oechem.OEDesignUnit or None
            The design unit or None if no design unit was found.
        """
        from openeye import oechem

        from ..modeling.OEModeling import prepare_structure
        from ..utils import LocalFileStorage, sha256_objects

        design_unit_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            sha256_objects([
                self.loop_db,
                structure,
                chain_id,
                alternate_location,
                has_ligand,
                ligand_name,
                model_loops_and_caps,
            ]),
            "oedu",
            self.cache_dir,
        )
        if not design_unit_path.is_file():
            logging.debug("Generating design unit ...")
            try:
                design_unit = prepare_structure(
                    structure,
                    loop_db=self.loop_db if model_loops_and_caps else None,
                    has_ligand=has_ligand,
                    ligand_name=ligand_name,
                    chain_id=chain_id,
                    alternate_location=alternate_location,
                    cap_termini=True if model_loops_and_caps else False
                )
            except ValueError:
                return None
            logging.debug("Writing design unit ...")
            oechem.OEWriteDesignUnit(str(design_unit_path), design_unit)
        # re-reading design unit helps proper capping of e.g. 2itz
        # TODO: revisit, report bug
        logging.debug("Reading design unit from file ...")
        design_unit = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(design_unit_path), design_unit)

        return design_unit

    @staticmethod
    def _get_components(
            design_unit: oechem.OEDesignUnit,
            chain_id: Union[str, None],
    ) -> Tuple[oechem.OEGraphMol(), oechem.OEGraphMol(), oechem.OEGraphMol()]:
        """
        Get protein, solvent and ligand components from an OpenEye design unit.

        Parameters
        ----------
        design_unit: oechem.OEDesignUnit
            The OpenEye design unit to extract components from.
        chain_id: str or None
            The chain ID of interest.

        Returns
        -------
        components: tuple of oechem.OEGraphMol, oechem.OEGraphMol and oechem.OEGraphMol
            OpenEye molecules holding protein, solvent and ligand.
        """
        from openeye import oechem

        from ..modeling.OEModeling import select_chain

        protein, solvent, ligand = oechem.OEGraphMol(), oechem.OEGraphMol(), oechem.OEGraphMol()

        logging.debug("Extracting molecular components ...")
        design_unit.GetProtein(protein)
        design_unit.GetSolvent(solvent)
        design_unit.GetLigand(ligand)

        if chain_id:  # some design units can contain multiple chains
            logging.debug("Selecting chain ...")
            protein = select_chain(protein, chain_id)
            try:
                solvent = select_chain(solvent, chain_id)
            except ValueError:
                logging.debug("No solvent atoms found in given chain.")
            try:
                ligand = select_chain(ligand, chain_id)
            except ValueError:
                logging.debug("No ligand atoms found in given chain.")

        # delete protein atoms with no name (found in prepared protein of 4ll0)
        for atom in protein.GetAtoms():
            if not atom.GetName().strip():
                logging.debug("Deleting unknown atom ...")
                protein.DeleteAtom(atom)

        # perceive residues to remove artifacts of other design units in the sequence of the protein
        # preserve certain properties to assure correct behavior of the pipeline
        preserved_info = (
                oechem.OEPreserveResInfo_ResidueNumber
                | oechem.OEPreserveResInfo_ResidueName
                | oechem.OEPreserveResInfo_AtomName
                | oechem.OEPreserveResInfo_ChainID
                | oechem.OEPreserveResInfo_HetAtom
                | oechem.OEPreserveResInfo_InsertCode
                | oechem.OEPreserveResInfo_AlternateLocation
        )
        oechem.OEPerceiveResidues(protein, preserved_info)
        oechem.OEPerceiveResidues(solvent, preserved_info)
        oechem.OEPerceiveResidues(ligand)

        logging.debug(
            "Number of component atoms: " +
            f"Protein - {protein.NumAtoms()}, " +
            f"Solvent - {solvent.NumAtoms()}, " +
            f"Ligand - {ligand.NumAtoms()}."
        )
        return protein, solvent, ligand

    def _process_protein(
            self,
            protein_structure: oechem.OEMolBase,
            amino_acid_sequence: AminoAcidSequence,
    ) -> oechem.OEMolBase:
        """
        Process a protein a structure according to the given amino acid sequence.

        Parameters
        ----------
        protein_structure: oechem.OEMolBase
            An OpenEye molecule holding the protein structure to process.
        amino_acid_sequence: core.sequences.AminoAcidSequence
            The amino acid sequence with associated metadata.

        Returns
        -------
        :oechem.OEMolBase
            An OpenEye molecule holding the processed protein structure.
        """
        from ..modeling.OEModeling import (
            read_molecules,
            assign_caps,
            apply_deletions,
            apply_insertions,
            apply_mutations,
            delete_clashing_sidechains,
            delete_partial_residues,
            delete_short_protein_segments,
            renumber_structure,
            write_molecules
        )
        from ..utils import LocalFileStorage, sha256_objects

        processed_protein_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            sha256_objects([self.loop_db, protein_structure, amino_acid_sequence]),
            "oeb",
            self.cache_dir,
        )
        if processed_protein_path.is_file():
            logging.debug("Reading processed protein from file ...")
            return read_molecules(processed_protein_path)[0]

        logging.debug(f"Deleting residues with clashing side chains ...")  # e.g. 2j5f, 4wd5
        protein_structure = delete_clashing_sidechains(protein_structure)

        logging.debug("Deleting residues with missing atoms ...")
        protein_structure = delete_partial_residues(protein_structure)

        logging.debug("Deleting loose protein segments ...")
        protein_structure = delete_short_protein_segments(protein_structure)

        logging.debug("Applying deletions to protein structure ...")
        protein_structure = apply_deletions(protein_structure, amino_acid_sequence)

        logging.debug("Deleting loose protein segments after applying deletions ...")
        protein_structure = delete_short_protein_segments(protein_structure)

        logging.debug("Applying mutations to protein structure ...")
        protein_structure = apply_mutations(protein_structure, amino_acid_sequence)

        logging.debug("Deleting loose protein segments after applying mutations ...")
        protein_structure = delete_short_protein_segments(protein_structure)

        logging.debug("Renumbering protein residues ...")
        residue_numbers = self._get_protein_residue_numbers(protein_structure, amino_acid_sequence)
        protein_structure = renumber_structure(protein_structure, residue_numbers)

        if self.loop_db:
            logging.debug("Applying insertions to protein structure ...")
            protein_structure = apply_insertions(protein_structure, amino_acid_sequence, self.loop_db)

        logging.debug("Checking protein structure sequence termini ...")
        real_termini = []
        if amino_acid_sequence.metadata["true_N_terminus"]:
            if amino_acid_sequence.metadata["begin"] == residue_numbers[0]:
                real_termini.append(residue_numbers[0])
        if amino_acid_sequence.metadata["true_C_terminus"]:
            if amino_acid_sequence.metadata["end"] == residue_numbers[-1]:
                real_termini.append(residue_numbers[-1])
        if len(real_termini) == 0:
            real_termini = None

        logging.debug(f"Assigning caps except for real termini {real_termini} ...")
        protein_structure = assign_caps(protein_structure, real_termini)

        logging.debug("Writing processed protein structure ...")
        write_molecules([protein_structure], processed_protein_path)

        return protein_structure

    @staticmethod
    def _get_protein_residue_numbers(
            protein_structure: oechem.OEMolBase,
            amino_acid_sequence: AminoAcidSequence
    ) -> List[int]:
        """
        Get the residue numbers of a protein structure according to given amino acid sequence.

        Parameters
        ----------
        protein_structure: oechem.OEMolBase
            The kinase domain structure.
        amino_acid_sequence: core.sequences.AminoAcidSequence
            The canonical kinase domain sequence.

        Returns
        -------
        residue_number: list of int
            A list of residue numbers according to the given amino acid sequence in the same order
            as the residues in the given protein structure.
        """
        from ..modeling.OEModeling import get_structure_sequence_alignment

        logging.debug("Aligning sequences ...")
        target_sequence, template_sequence = get_structure_sequence_alignment(
            protein_structure, amino_acid_sequence)
        logging.debug(f"Template sequence:\n{template_sequence}")
        logging.debug(f"Target sequence:\n{target_sequence}")

        logging.debug("Generating residue numbers ...")
        residue_numbers = []
        residue_number = amino_acid_sequence.metadata["begin"]
        for template_sequence_residue, target_sequence_residue in zip(
                template_sequence, target_sequence
        ):
            if template_sequence_residue != "-":
                if target_sequence_residue != "-":
                    residue_numbers.append(residue_number)
                residue_number += 1
            else:
                # I doubt this this will ever happen in the current implementation
                text = (
                    "Cannot generate residue IDs. The given protein structure contain residues "
                    "that are not part of the canoical sequence from UniProt."
                )
                logging.debug("Exception: " + text)
                raise ValueError(text)

        return residue_numbers

    def _assemble_components(
        self,
        protein: oechem.OEMolBase,
        solvent: oechem.OEMolBase,
        ligand: Union[oechem.OEMolBase, None] = None
    ) -> oechem.OEMolBase:
        """
        Assemble components of a solvated protein-ligand complex into a single OpenEye molecule.

        Parameters
        ----------
        protein: oechem.OEMolBase
            An OpenEye molecule holding the protein of interest.
        solvent: oechem.OEMolBase
            An OpenEye molecule holding the solvent of interest.
        ligand: oechem.OEMolBase or None, default=None
            An OpenEye molecule holding the ligand of interest if given.

        Returns
        -------
        assembled_components: oechem.OEMolBase
            An OpenEye molecule holding protein, solvent and ligand if given.
        """
        from openeye import oechem

        from ..modeling.OEModeling import update_residue_identifiers

        assembled_components = oechem.OEGraphMol()

        logging.debug("Adding protein ...")
        oechem.OEAddMols(assembled_components, protein)

        if ligand:
            logging.debug("Renaming ligand ...")
            for atom in ligand.GetAtoms():
                oeresidue = oechem.OEAtomGetResidue(atom)
                oeresidue.SetName("LIG")
                oechem.OEAtomSetResidue(atom, oeresidue)

            logging.debug("Adding ligand ...")
            oechem.OEAddMols(assembled_components, ligand)

        logging.debug("Adding water molecules ...")
        filtered_solvent = self._remove_clashing_water(solvent, ligand, protein)
        oechem.OEAddMols(assembled_components, filtered_solvent)

        logging.debug("Updating hydrogen positions of assembled components ...")
        options = oechem.OEPlaceHydrogensOptions()  # keep protonation state from docking
        predicate = oechem.OEAtomMatchResidue(["LIG:.*:.*:.*:.*"])
        options.SetBypassPredicate(predicate)
        oechem.OEPlaceHydrogens(assembled_components, options)
        # keep tyrosine protonated, e.g. 6tg1 chain B
        predicate = oechem.OEAndAtom(
            oechem.OEAtomMatchResidue(["TYR:.*:.*:.*:.*"]),
            oechem.OEHasFormalCharge(-1)
        )
        for atom in assembled_components.GetAtoms(predicate):
            if atom.GetName().strip() == "OH":
                atom.SetFormalCharge(0)
                atom.SetImplicitHCount(1)
        oechem.OEAddExplicitHydrogens(assembled_components)

        logging.debug("Updating residue identifiers ...")
        assembled_components = update_residue_identifiers(assembled_components)

        return assembled_components

    @staticmethod
    def _remove_clashing_water(
        solvent: oechem.OEMolBase,
        ligand: Union[oechem.OEMolBase, None],
        protein: oechem.OEMolBase
    ) -> oechem.OEGraphMol:
        """
        Remove water molecules clashing with a ligand or newly modeled protein residues.

        Parameters
        ----------
        solvent: oechem.OEGraphMol
            An OpenEye molecule holding the water molecules.
        ligand: oechem.OEGraphMol or None
            An OpenEye molecule holding the ligand or None.
        protein: oechem.OEGraphMol
            An OpenEye molecule holding the protein.

        Returns
        -------
        :oechem.OEGraphMol
            An OpenEye molecule holding water molecules not clashing with the ligand or newly
            modeled protein residues.
        """
        from openeye import oechem, oespruce
        from scipy.spatial import cKDTree

        from ..modeling.OEModeling import get_atom_coordinates, split_molecule_components

        if ligand is not None:
            ligand_heavy_atoms = oechem.OEGraphMol()
            oechem.OESubsetMol(
                ligand_heavy_atoms,
                ligand,
                oechem.OEIsHeavy()
            )
            ligand_heavy_atom_coordinates = get_atom_coordinates(ligand_heavy_atoms)
            ligand_heavy_atoms_tree = cKDTree(ligand_heavy_atom_coordinates)

        modeled_heavy_atoms = oechem.OEGraphMol()
        oechem.OESubsetMol(
            modeled_heavy_atoms,
            protein,
            oechem.OEAndAtom(
                oespruce.OEIsModeledAtom(),
                oechem.OEIsHeavy()
            )
        )
        modeled_heavy_atoms_tree = None
        if modeled_heavy_atoms.NumAtoms() > 0:
            modeled_heavy_atom_coordinates = get_atom_coordinates(modeled_heavy_atoms)
            modeled_heavy_atoms_tree = cKDTree(modeled_heavy_atom_coordinates)

        filtered_solvent = oechem.OEGraphMol()
        waters = split_molecule_components(solvent)
        # iterate over water molecules and check for clashes and ambiguous water molecules
        for water in waters:
            try:
                water_oxygen_atom = water.GetAtoms(oechem.OEIsOxygen()).next()
            except StopIteration:
                # experienced lonely water hydrogens for 2v7a after mutating PTR393 to TYR
                logging.debug("Removing water molecule without oxygen!")
                continue
            # experienced problems when preparing 4pmp
            # making design units generated clashing waters that were not protonatable
            # TODO: revisit this behavior
            if oechem.OEAtomGetResidue(water_oxygen_atom).GetInsertCode() != " ":
                logging.debug("Removing ambiguous water molecule!")
                continue
            water_oxygen_coordinates = water.GetCoords()[water_oxygen_atom.GetIdx()]
            # check for clashes with newly placed ligand
            if ligand is not None:
                clashes = ligand_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Removing water molecule clashing with ligand atoms!")
                    continue
            # check for clashes with newly modeled protein residues
            if modeled_heavy_atoms_tree:
                clashes = modeled_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Removing water molecule clashing with modeled atoms!")
                    continue
            # water molecule is not clashy, add to filtered solvent
            oechem.OEAddMols(filtered_solvent, water)

        return filtered_solvent

    def _update_pdb_header(
        self,
        structure: oechem.OEMolBase,
        protein_name: str,
        ligand_name: [str, None] = None,
        other_pdb_header_info: Union[None, Iterable[Tuple[str, str]]] = None
    ) -> oechem.OEMolBase:
        """
        Stores information about Featurizer, protein and ligand in the PDB header COMPND section in the
        given OpenEye molecule.

        Parameters
        ----------
        structure: oechem.OEMolBase
            An OpenEye molecule.
        protein_name: str
            The name of the protein.
        ligand_name: str or None, default=None
            The name of the ligand if present.
        other_pdb_header_info: None or iterable of tuple of str
            Tuples with information that should be saved in the PDB header. Each tuple consists of two strings,
            i.e., the PDB header section (e.g. COMPND) and the respective information.

        Returns
        -------
        :oechem.OEMolBase
            The OpenEye molecule containing the updated PDB header.
        """
        from openeye import oechem

        oechem.OEClearPDBData(structure)
        oechem.OESetPDBData(structure, "COMPND", f"\tFeaturizer: {self.__class__.__name__}")
        oechem.OEAddPDBData(structure, "COMPND", f"\tProtein: {protein_name}")
        if ligand_name:
            oechem.OEAddPDBData(structure, "COMPND", f"\tLigand: {ligand_name}")
        if other_pdb_header_info is not None:
            for section, information in other_pdb_header_info:
                oechem.OEAddPDBData(structure, section, information)

        return structure

    def _write_results(
        self,
        structure: oechem.OEMolBase,
        protein_name: str,
        ligand_name: Union[str, None] = None,
     ) -> Path:
        """
        Write the results from the Featurizer and retrieve the paths to protein or complex if a
        ligand is present.

        Parameters
        ----------
        structure: oechem.OEMolBase
            The OpenEye molecule holding the featurized system.
        protein_name: str
            The name of the protein.
        ligand_name: str or None, default=None
            The name of the ligand if present.

        Returns
        -------
        :Path
            Path to prepared protein or complex if ligand is present.
        """
        from openeye import oechem

        from ..modeling.OEModeling import write_molecules, remove_non_protein
        from ..utils import LocalFileStorage

        if self.output_dir:
            if ligand_name:
                logging.debug("Writing protein ligand complex ...")
                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([structure], complex_path)

                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([structure], complex_path)

                logging.debug("Splitting components")
                solvated_protein = remove_non_protein(structure, remove_water=False)
                split_options = oechem.OESplitMolComplexOptions()
                ligand = list(oechem.OEGetMolComplexComponents(
                    structure, split_options, split_options.GetLigandFilter())
                )[0]

                logging.debug("Writing protein ...")
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_protein",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([solvated_protein], protein_path)

                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_protein",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([solvated_protein], protein_path)

                logging.debug("Writing ligand ...")
                ligand_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_ligand",
                    "sdf",
                    self.output_dir,
                )
                write_molecules([ligand], ligand_path)

                return complex_path
            else:
                logging.debug("Writing protein ...")
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([structure], protein_path)

                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([structure], protein_path)

                return protein_path
        else:
            if ligand_name:
                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "pdb",
                )
                write_molecules([structure], complex_path)

                return complex_path
            else:
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "pdb",
                )
                write_molecules([structure], protein_path)

                return protein_path
