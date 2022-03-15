"""
Base classes for ``DatasetProvider``-like objects
"""

import logging
from typing import Iterable
from collections import defaultdict
from urllib.request import urlopen
import shutil
from pathlib import Path
import os

import numpy as np
import pandas as pd
import awkward as ak

from ..core.measurements import BaseMeasurement
from ..features.core import BaseFeaturizer
from ..utils import APPDIR

logger = logging.getLogger(__name__)


class BaseDatasetProvider(object):
    """
    API specification for dataset providers
    """

    @classmethod
    def from_source(cls, path_or_url=None, **kwargs):
        """
        Parse CSV/raw files to object model.
        """
        raise NotImplementedError

    def observation_model(self, backend="pytorch"):
        raise NotImplementedError

    @property
    def systems(self):
        raise NotImplementedError

    @property
    def measurement_type(self):
        raise NotImplementedError

    def measurements_as_array(self, reduce=np.mean):
        raise NotImplementedError

    def measurements_by_group(self):
        raise NotImplementedError

    @property
    def conditions(self):
        raise NotImplementedError

    def featurize(self, *featurizers: Iterable[BaseFeaturizer]):
        raise NotImplementedError

    def featurized_systems(self, key="last"):
        raise NotImplementedError

    def to_dataframe(self, *args, **kwargs):
        raise NotImplementedError

    def to_pytorch(self, **kwargs):
        raise NotImplementedError

    def to_tensorflow(self, *args, **kwargs):
        raise NotImplementedError

    def to_numpy(self, *args, **kwargs):
        raise NotImplementedError


class DatasetProvider(BaseDatasetProvider):

    """
    Base object for all DatasetProvider classes.

    Parameters
    ----------
    measurements: list of BaseMeasurement
        A DatasetProvider holds a list of ``kinoml.core.measurements.BaseMeasurement``
        objects (or any of its subclasses). They must be of the same type!
    metadata: dict
        Extra information for provenance.

    Note
    ----
    All measurements must be of the same type! If they are not, consider
    using ``MultiDatasetProvider`` instead.
    """

    _raw_data = None

    def __init__(self, measurements: Iterable[BaseMeasurement], metadata: dict = None):
        types = {type(measurement) for measurement in measurements}
        assert (
            len(types) == 1
        ), f"Dataset providers can only allow one type of measurement! You provided: {types}"
        self.measurements = measurements
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return self.__class__(self.measurements[subscript])
        else:
            return self.measurements[subscript]

    def __repr__(self) -> str:
        components = defaultdict(set)
        for s in self.systems:
            for c in s.components:
                components[type(c).__name__].add(c.name)
        components_str = ", ".join([f"{k}={len(v)}" for k, v in components.items()])
        return (
            f"<{self.__class__.__name__} with "
            f"{len(self.measurements)} {self.measurement_type.__name__} measurements "
            f"and {len(self.systems)} systems ({components_str})>"
        )

    @classmethod
    def from_source(cls, path_or_url=None, **kwargs):
        """
        Parse CSV/raw file to object model. This method is responsible of generating
        the objects for ``self.measurements``, if relevant. Additional kwargs will be
        passed to ``__init__``.

        You must define this in your subclass.
        """
        raise NotImplementedError

    def featurize(self, featurizer: BaseFeaturizer):
        """
        Given a collection of ``kinoml.features.core.BaseFeaturizers``, apply them
        to the systems present in the ``self.measurements``.

        Parameters
        ----------
        featurizer : BaseFeaturizer
            Featurization scheme that will be applied to the systems,
            in a stacked way.

        Note
        ----
        TODO:
            * Will the systems be properly featurized with Dask?
        """
        systems = self.systems
        featurizer.supports(next(iter(systems)), raise_errors=True)
        featurizer.featurize(tuple(systems))
        n_invalid = len(
            [system for system in systems if featurizer.name not in system.featurizations.keys()]
        )
        if n_invalid > 0:
            logger.warning(f"There were {n_invalid} systems that could not be featurized!")
        self._post_featurize(featurizer)

        return systems

    def _post_featurize(self, featurizer: BaseFeaturizer):
        """
        Remove measurements with systems, that were not successfully featurized.

        Parameters
        ----------
        featurizer: BaseFeaturizer
            The used featurizer.
        """
        self.measurements = [
            measurement
            for measurement in self.measurements
            if featurizer.name in measurement.system.featurizations.keys()
        ]

    def featurized_systems(self, key="last", clear_after=False):
        """
        Return the ``key`` featurized objects from all systems.
        """
        results = tuple(ms.system.featurizations[key] for ms in self.measurements)
        if clear_after:
            for s in self.systems:
                s.featurizations.clear()
        return results

    def _to_dataset(self, style="pytorch"):
        """
        Generate a clean <style>.data.Dataset object for further steps
        in the pipeline (model building, etc).

        .. warning::

            This step is lossy because the resulting objects will no longer
            hold chemical data. Operations depending on such information,
            must be performed first.

        Examples
        --------
        >>> provider = DatasetProvider()
        >>> provider.featurize()  # optional
        >>> splitter = TimeSplitter()
        >>> split_indices = splitter.split(provider.data)
        >>> dataset = provider.to_dataset("pytorch")  # .featurize() under the hood
        >>> X_train, X_test, y_train, y_test = train_test_split(dataset, split_indices)
        """
        raise NotImplementedError

    def to_dataframe(self, *args, **kwargs):
        """
        Generates a ``pandas.DataFrame`` containing information on the systems
        and their measurements

        Returns
        -------
        pandas.DataFrame
        """
        if not self.systems:
            return pd.DataFrame()
        columns = ["Systems", "n_components", self.measurements[0].__class__.__name__]
        records = [
            (
                measurement.system.name,
                len(measurement.system.components),
                measurement.values.mean(),
            )
            for measurement in self.measurements
        ]

        return pd.DataFrame.from_records(records, columns=columns)

    def to_pytorch(self, featurizer=None, **kwargs):
        """
        Export dataset to a PyTorch-compatible object, via adapters
        found in ``kinoml.torch_datasets``.
        """
        from .torch_datasets import TorchDataset, PrefeaturizedTorchDataset

        if featurizer is not None:
            return TorchDataset(
                [ms.system for ms in self.measurements],
                self.measurements_as_array(**kwargs),
                featurizer=featurizer,
                observation_model=self.observation_model(backend="pytorch"),
            )
        # else
        return PrefeaturizedTorchDataset(
            self.featurized_systems(),
            self.measurements_as_array(**kwargs),
            observation_model=self.observation_model(backend="pytorch"),
        )

    def to_xgboost(self, **kwargs):
        """
        Export dataset to a ``DMatrix`` object, native to the XGBoost framework
        """
        from xgboost import DMatrix

        dmatrix = DMatrix(self.to_numpy(**kwargs))
        ## TODO: Uncomment when XGB observation models are implemented
        # dmatrix.observation_model = self.observation_model(backend="xgboost", loss="mse")
        return dmatrix

    def to_tensorflow(self, *args, **kwargs):
        raise NotImplementedError

    def to_numpy(self, featurization_key="last", y_dtype="float32", **kwargs):
        """
        Export dataset to a tuple of two Numpy arrays of same shape:

        * ``X``: the featurized systems
        * ``y``: the measurements values (must be the same measurement type)

        Parameters
        ----------
        featurization_key : str, optional="last"
            Which featurization present in the systems will be taken
            to build the ``X`` array. Usually, ``last`` as provided
            by a ``Pipeline`` object.
        y_dtype, np.dtype or str, optional="float32"
            Coerce Y array to this dtype
        kwargs : optional,
            Dict that will be forwarded to ``.measurements_as_array``,
            which will build the ``y`` array.

        Returns
        -------
        2-tuple of np.array
            X, y

        Note
        ----
        This exporter assumes that each System is featurized as a single
        tensor with homogeneous shape throughout the system collection.
        If this does not hold true for your current featurization
        scheme, consider using ``.to_dict_of_arrays`` instead.
        """
        X = np.asarray(self.featurized_systems(key=featurization_key))
        y = self.measurements_as_array(dtype=y_dtype, **kwargs)
        assert (
            X.shape[0] == y.shape[0]
        ), f"# of X ({X.shape[0]}) and y ({y.shape[0]}) do not match!"
        return X, y

    def to_dict_of_arrays(
        self, featurization_key="last", y_dtype="float32", _initial_system_index=0
    ) -> dict:
        """
        Export dataset to a dict-like object, compatible
        with ``DictOfArrays`` and NPZ files.

        The idea is to provide unique keys for each system
        and their features, following the syntax
        ``X_s{int}_v{int}``.

        This object is useful when the features for each system
        have different shapes and/or dimensionality and cannot
        be concatenated in a single homogeneous array

        Parameters
        ----------
        featurization_key : Hashable, optional="last"
            Which key to access in each ``System.featurizations`` dict
        y_dtype : np.dtype or str, optional="float32"
            Which kind of dtype to use for the ``y`` array
        _initial_system_index : int, optional=0
            PRIVATE. Start counting systems in ``X_s{int}`` with this value.

        Returns
        -------
        dict[str, array]
            A dictionary that maps ``str`` keys to array-like
            objects. Depending on the featurization scheme, keys
            can be:

            1. All systems are featurized as an array and they share the same shape
               -> ``X, y``

            2. All N systems are featurized as an array but they do NOT share the same shape
               -> ``X_s0_, X_s1_, ..., X_sN_``

            3. All N systems are featurized as a M-tuple of arrays (shape irrelevant)
               -> ``X_s0_a0_, X_s0_a1_, X_s1_a0_, X_s1_a1_, ..., X_sN_aM_``


        Note
        ----
        The X keys have a trailing underscore on purpose. Otherwise, filtering
        keys out of the dictionary by index can be deceivingly slow. For example,
        filtering for the first system (s1) with ``key.startswith("X_s1")`` will
        also select for X_s1, X_s10, X_s11... Hence, we filter with ``X_s{int}_``.
        """
        featurized = self.featurized_systems(key=featurization_key)
        nsystems = len(featurized)
        dict_of_arrays = {}
        y = self.measurements_as_array(dtype=y_dtype)
        assert (
            nsystems == y.shape[0]
        ), f"# of systems ({nsystems}) and measurements {y.shape[0]} do not match!"
        # See which kind of feature object we are handling
        if isinstance(featurized[0], np.ndarray):
            # each system _is_ an array already
            if all(featurized[0].shape == f.shape for f in featurized[1:]):
                # all arrays are the same shape, we can return a unified X!
                dict_of_arrays["X"] = np.asarray(featurized)
            else:
                # each system might have different shapes, we need separate X
                for i, feature in enumerate(featurized, start=_initial_system_index):
                    key = f"X_s{i}_"
                    dict_of_arrays[key] = feature
        elif isinstance(featurized[0], (list, tuple)) and isinstance(featurized[0][0], np.ndarray):
            # each system has a list of arrays
            for i, system in enumerate(featurized, start=_initial_system_index):
                for j, feature in enumerate(system):
                    key = f"X_s{i}_a{j}_"
                    dict_of_arrays[key] = feature
        else:
            raise ValueError(
                "Current featurization scheme is not supported! "
                "Features must be either: same-shape arrays, different-shape arrays, "
                "or a list/tuple of arrays (irrelevant shape). Peek at first element:\n"
                f"{featurized[0]}"
            )

        dict_of_arrays["y"] = y

        return dict_of_arrays

    def to_awkward(
        self,
        featurization_key="last",
        y_dtype="float32",
        clear_after=False,
    ):
        """
        Creates an awkward array out of the featurized systems
        and the associated measurements.

        Returns
        -------
        awkward array

        Notes
        -----
        Awkward Array is a library for nested, variable-sized data,
        including arbitrary-length lists, records, mixed types,
        and missing data, using NumPy-like idioms.

        Arrays are dynamically typed, but operations on
        them are compiled and fast. Their behavior coincides
        with NumPy when array dimensions are regular
        and generalizes when they’re not.
        """
        features = self.featurized_systems(key=featurization_key, clear_after=clear_after)

        # Features is a list of systems (s) and their features (f): [(s0f0, s0f1), (s1f0, s1f1)...]
        # We are going to iterate over columns: (s0f0, s1f0... snf0), (s0f1, s1f1, ..., snf1)
        # The result is that X will contain the an array of f0, then an array for f1... etc.
        X = [ak.from_iter(subX) for subX in zip(*features)]
        y = ak.from_numpy(self.measurements_as_array(dtype=y_dtype))
        return X, y

    def observation_model(self, **kwargs):
        """
        Draft implementation of a modular observation model, based on individual contributions
        from different measurement types.
        """
        return self.measurement_type.observation_model(**kwargs)

    def loss_adapter(self, **kwargs):
        """
        Observation model plus loss function, wrapped in a single callable. Return types are
        backend-dependent.
        """
        return self.measurement_type.loss_adapter(**kwargs)

    @property
    def systems(self):
        return list({ms.system for ms in self.measurements})

    @property
    def measurement_type(self):
        return type(self.measurements[0])

    def measurements_as_array(self, reduce=np.mean, dtype="float32"):
        result = np.empty(len(self.measurements), dtype=dtype)
        for i, measurement in enumerate(self.measurements):
            if measurement.values.shape[0] > 1:
                result[i] = reduce(measurement.values)
            else:
                result[i] = measurement.values[0]
        return result

    def split_by_groups(self) -> dict:
        """
        If a ``kinoml.datasets.groups`` class has been applied to this instance,
        this method will create more DatasetProvider instances, one per group.

        Returns
        -------
        dict
            Maps group key to sub-datasets
        """
        groups = defaultdict(list)
        for measurement in self.measurements:
            groups[measurement.group].append(measurement)

        datasets = {}
        for key, measurements in groups.items():
            datasets[key] = type(self)(measurements)
        return datasets

    @property
    def conditions(self) -> set:
        return {ms.conditions for ms in self.measurements}

    @classmethod
    def _download_to_cache_or_retrieve(cls, path_or_url) -> str:
        """
        Helper function to either download files to the usercache, or
        retrieve an already cached copy.

        Parameters
        ----------
        path_or_url : str or Path-like
            File path or URL pointing to the required file

        Returns
        -------
        str
            If provided argument is a file, the same path, right away
            If it was a URL, it will be the (downloaded) cached file path
        """
        if os.path.isfile(path_or_url):
            return str(path_or_url)
        filename = os.path.basename(path_or_url)
        cached_path = Path(APPDIR.user_cache_dir) / cls.__name__ / filename
        if not cached_path.is_file():  # file is not available on user cache
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(path_or_url) as f, open(cached_path, "wb") as dest:
                shutil.copyfileobj(f, dest)
        return str(cached_path)


class MultiDatasetProvider(DatasetProvider):
    """
    Adapter class that is able to expose a DatasetProvider-like
    interface to a collection of Measurements of different types.

    The different types are split into individual DatasetProvider
    objects, stored under ``.providers``.

    The rest of the API works around that list to provide
    similar functionality as the original, single-type DatasetProvider,
    but in plural.

    Parameters
    ----------
    measurements: list of BaseMeasurement
        A MultiDatasetProvider holds a list of
        ``kinoml.core.measurements.BaseMeasurement`` objects
        (or any of its subclasses). Unlike ``DatasetProvider``,
        the measurements here can be of different types, but they
        will be grouped together in different sub-datasets.
    """

    def __init__(self, measurements: Iterable[BaseMeasurement], metadata: dict = None):
        by_type = defaultdict(list)
        for measurement in measurements:
            by_type[type(measurement)].append(measurement)

        providers = []
        for typed_measurements in by_type.values():
            if typed_measurements:
                providers.append(DatasetProvider(typed_measurements))

        self.providers = providers
        self.metadata = metadata or {}

    def _post_featurize(self, featurizer: BaseFeaturizer):
        """
        Remove measurements with systems, that were not successfully featurized.

        Parameters
        ----------
        featurizer: BaseFeaturizer
            The used featurizer.
        """
        for provider in self.providers:
            provider.measurements = [
                measurement
                for measurement in provider.measurements
                if featurizer.name in measurement.system.featurizations.keys()
            ]

    def observation_models(self, **kwargs):
        """
        List of observation models present in this dataset,
        one per provider (measurement type)
        """
        return [p.observation_model(**kwargs) for p in self.providers]

    def loss_adapters(self, **kwargs):
        """
        List of observation models present in this dataset,
        one per provider (measurement type)
        """
        return [p.loss_adapter(**kwargs) for p in self.providers]

    def observation_model(self, **kwargs):
        raise NotImplementedError(f"{type(self)} must use `.observation_models()` (plural)")

    def loss_adapter(self, **kwargs):
        raise NotImplementedError(f"{type(self)} must use `.loss_adapters()` (plural)")

    @property
    def measurements(self):
        """
        Flattened list of all measurements present across all providers.

        Use ``.indices_by_provider()`` to obtain the corresponding slices
        to each provider.
        """
        return [ms for prov in self.providers for ms in prov.measurements]

    def indices_by_provider(self) -> dict:
        """
        Return a dict mapping each ``provider`` type to their
        correlative indices in a hypothetically concatenated
        dataset.

        For example, if a ``MultiDatasetProvider`` contains
        50 measurements of type A, and 25 measurements of
        type B, this would return ``{"A": slice(0, 50), "B": slice(50, 75)}``.

        Note
        ----
        ``slice`` objects can be passed directly to item access syntax, like
        ``list[slice(a, b)]``.
        """
        indices = {}
        offset = 0
        for p in self.providers:
            indices[p.measurement_type] = slice(offset, offset + len(p.measurements))
            offset += len(p.measurements)
        return indices

    def to_dataframe(self, *args, **kwargs):
        """
        Concatenate all the providers into a single DataFrame for easier visualization.

        Check ``DatasetProvider.to_dataframe()`` for more details.
        """
        columns = ["Systems", "n_components", "Measurement", "MeasurementType"]
        records = []
        for provider in self.providers:
            measurement_type = provider.measurement_type.__name__
            for measurement in provider.measurements:
                records.append(
                    (
                        measurement.system.name,
                        len(measurement.system.components),
                        measurement.values.mean(),
                        measurement_type,
                    )
                )

        return pd.DataFrame.from_records(records, columns=columns)

    def to_numpy(self, **kwargs):
        """
        List of Numpy-native arrays, as generated by each ``provider.to_numpy(...)``
        method. Check ``DatasetProvider.to_numpy`` docstring for more details.
        """
        return [p.to_numpy(**kwargs) for p in self.providers]

    def to_pytorch(self, **kwargs):
        """
        List of Numpy-native arrays, as generated by each ``provider.to_pytorch(...)``
        method. Check ``DatasetProvider.to_pytorch`` docstring for more details.
        """
        return [p.to_pytorch(**kwargs) for p in self.providers]

    def to_xgboost(self, **kwargs):
        """
        List of Numpy-native arrays, as generated by each ``provider.to_xgboost(...)``
        method. Check ``DatasetProvider.to_xgboost`` docstring for more details.
        """
        return [p.to_xgboost(**kwargs) for p in self.providers]

    def to_dict_of_arrays(self, **kwargs) -> dict:
        """
        Will generate a dictionary of str: np.ndarray. System indices
        will be accumulated.
        """
        all_arrays = {}
        system_index = 0
        for p in self.providers:
            arrays = p.to_dict_of_arrays(_initial_system_index=system_index, **kwargs)
            system_index += len(p)
            for key, arr in arrays.items():
                if key in all_arrays:
                    all_arrays[key] = np.concatenate([all_arrays[key], arr])
                else:
                    all_arrays[key] = arr
        return all_arrays

    def to_awkward(self, **kwargs):
        """
        See ``DatasetProvider.to_awkward()``. ``X`` and ``y`` will
        be concatenated along axis=0 (one provider after another)
        """
        all_X = []
        all_y = []
        for p in self.providers:
            X, y = p.to_awkward(**kwargs)
            all_X.append(X)
            all_y.append(y)

        return ak.concatenate(all_X), ak.concatenate(all_y)

    def __repr__(self) -> str:
        measurements = []
        for p in self.providers:
            measurements.append(f"{p.measurement_type.__name__}={len(p)}")
        components = defaultdict(set)
        for s in self.systems:
            for c in s.components:
                components[type(c).__name__].add(c.name)
        components_str = ", ".join([f"{k}={len(v)}" for k, v in components.items()])
        return (
            f"<{self.__class__.__name__} with "
            f"{len(self)} measurements ({', '.join(measurements)}), "
            f"and {len(self.systems)} systems ({components_str})>"
        )
