"""
Helper classes to convert between DatasetProvider objects and
Dataset-like objects native to the PyTorch ecosystem
"""
from collections import defaultdict
from typing import List
from pathlib import Path

import numpy as np
import awkward as ak
import torch
from torch.utils.data import Dataset as _NativeTorchDataset, DataLoader as _DataLoader
from tqdm.auto import tqdm

from ..core.measurements import null_observation_model as _null_observation_model

# Disable false positive lint with torch.tensor
# see https://github.com/pytorch/pytorch/issues/24807
# pylint: disable=not-callable


class PrefeaturizedTorchDataset(_NativeTorchDataset):
    """
    Exposes the ``X``, ``y`` (systems and measurements, respectively)
    arrays exported by ``DatasetProvider`` using the API expected
    by Torch DataLoaders.

    Parameters
    ----------
    systems : array-like
        X vectors, as exported from featurized systems in DatasetProvider
    measurements : array-like
        y vectors, as exported from the measurement values contained in a
        DatasetProvider
    observation_model : callable, optional
        A function that adapts the predicted ``y`` to the observed ``y``
        values. Useful to combine measurement types in the same model, if
        they are mathematically related. Normally provided by the
        ``Measurement`` type class.
    """

    def __init__(
        self, systems, measurements, observation_model: callable = _null_observation_model,
    ):
        assert len(systems) == len(measurements), "Systems and Measurements must match in size!"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # note we are using as_tensor to _avoid_ copies if possible
        self.systems = systems
        self.measurements = measurements
        self.observation_model = observation_model

    def __getitem__(self, index):
        X = torch.tensor(self.systems[index], device=self.device, dtype=torch.float)
        y = torch.tensor(self.measurements[index], device=self.device, dtype=torch.float)
        return X, y

    def __len__(self):
        return len(self.systems)

    def as_dataloader(self, **kwargs):
        """
        Build a PyTorch DataLoader view of this Dataset
        """
        return _DataLoader(dataset=self, **kwargs)

    def estimate_input_size(self) -> int:
        """
        Estimate the input size for a model, using
        the first dimension of the ``X`` vector shape.
        """
        return self.systems[0].shape


class TorchDataset(PrefeaturizedTorchDataset):
    """
    Same purpose as ``PrefeaturizedTorchDataset``, but
    instead of taking arrays in, it takes the non-featurized
    ``System`` and ``Measurement``objects, and applies a
    ``featurizer`` on the fly upon access (e.g. during training).

    Parameters
    ----------
    systems : list of kinoml.core.systems.System
    measurements : list of kinoml.core.measurements.BaseMeasurement
    featurizer : callable
        A function that takes a ``System`` and returns an array-like
        object.
    observation_model : callable, optional
        A function that adapts the predicted ``y`` to the observed ``y``
        values. Useful to combine measurement types in the same model, if
        they are mathematically related. Normally provided by the
        ``Measurement`` type class.
    """

    def __init__(
        self,
        systems,
        measurements,
        featurizer,
        observation_model: callable = _null_observation_model,
    ):
        super().__init__(systems, measurements, observation_model=observation_model)
        if featurizer is None:
            raise ValueError("TorchDataset requires `featurizer` keyword argument!")
        self.featurizer = featurizer

    def estimate_input_size(self):
        return self.featurizer(self.systems[0]).featurizations[self.featurizer.name].shape

    def __getitem__(self, index):
        """
        In this case, the DatasetProvider is passing System objects that will
        be featurized (and memoized) upon access only.
        """
        # TODO: featurize y?

        X = torch.tensor(
            self.featurizer(self.systems[index]).featurizations[self.featurizer.name],
            device=self.device,
            dtype=torch.float,
            requires_grad=True,
        )
        y = torch.tensor(
            self.measurements[index], device=self.device, requires_grad=True, dtype=torch.float,
        )
        return X, y


class XyTorchDataset(_NativeTorchDataset):
    """
    Simple Torch Dataset adaptor where X and y are homogeneous tensors.
    All systems have the shape.

    Parameters
    ----------
    X, y : arraylike
        Featurized systems and their measurements
    indices : dict of array selectors
        It will only accept train, train/test or train/test/val keys.
    """

    def __init__(self, X, y, indices=None):
        assert X.shape[0] == y.shape[0], "X and y must have the same number of elements"
        self.data_X = torch.as_tensor(X)
        self.data_y = torch.as_tensor(y)

        self.indices = indices or {"train": True}
        if len(self.indices) == 3:
            assert sorted(indices) == ["test", "train", "val"]
        elif len(self.indices) == 2:
            assert sorted(indices) == ["test", "train"]
        elif len(self.indices) == 1:
            assert sorted(indices) == ["test"]
        else:
            raise ValueError(
                "`indices` can only contain up to three keys: train, test and val, "
                f"but you provided `{list(self.indices.keys())}`"
            )

    @classmethod
    def from_npz(cls, path):
        """
        Load ``X`` and ``y`` arrays from a NPZ file present in disk.
        These files must expose at least two keys: ``X`` and ``y``.
        It can also contain three more: ``idx_train``, ``idx_test``
        and ``idx_val``, which correspond to the indices of the
        training, test and validation subsets.

        Parameters
        ----------
        npz : str
            Path to a NPZ file with the keys exposed above.
        """
        data = np.load(path)
        X = torch.as_tensor(data["X"])
        y = torch.as_tensor(data["y"])
        if "idx_train" in data:
            indices = {
                key[4:]: data[key] for key in ["idx_train", "idx_test", "idx_val"] if key in data
            }
        else:
            indices = {"train": True}
        return cls(X, y, indices=indices)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]

    def __len__(self):
        return self.data_X.shape[0]

    def input_size(self):
        # Optional for some models!
        return self.data_X.shape[1]


class MultiXTorchDataset(_NativeTorchDataset):
    """
    This class is able to load NPZ files into a ``torch.Dataset`` compliant
    object.

    It assumes the following things. If each system is characterized with
    a single tensor:

    - The X tensors can be of the same shape. In that case, the NPZ file
      only has a single ``X`` key, preloaded and accessible via ``.data_X`.
      When queried, it returns a view to the ``torch.tensor`` object.
    - The X tensors have different shape. In that case, the keys of the NPZ
      follow the ``X_s{int}`` syntax. When queried, it returns a list of
      ``torch.tensor`` objects.

    If each system is characterized with more than one tensor:

    - The NPZ keys follow the ``X_s{int}_a{int}`` syntax. When queried, it
      returns a list of tuples of ``torch.tensor`` objects.

    No matter the structure of ``X``, ``y`` is assumed to be a homogeneous
    tensor, and it will always be returned as a view to the underlying
    ``torch.tensor`` object.

    Additionally, the NPZ file might contain ``idx_train``, ``idx_test`` (and ``idx_val``)
    arrays, specifying indices for the train / test / validation split. If provided,
    they will be stored under an ``.indices`` dict.

    Parameters
    ----------
    dict_of_arrays : dict of np.ndarray
        See above.
    indices : dict of np.ndarray

    Notes
    -----

    - This object is better paired with the output of ``DatasetProvider.to_dict_of_arrays``.

    """

    def __init__(self, dict_of_arrays, indices=None):
        self._data = dict_of_arrays
        self.data_y = torch.tensor(dict_of_arrays["y"])
        if self.is_single_X():
            self.data_X = torch.tensor(dict_of_arrays["X"])
        else:
            self.data_X = None

        self.shape_X = self._shape_X()
        self.shape_y = self.data_y.shape

        self.indices = indices or {"train": True}
        if len(self.indices) == 3:
            assert sorted(indices) == ["test", "train", "val"]
        elif len(self.indices) == 2:
            assert sorted(indices) == ["test", "train"]
        elif len(self.indices) == 1:
            assert sorted(indices) == ["test"]
        else:
            raise ValueError(
                "`indices` can only contain up to three keys: train, test and val, "
                f"but you provided `{list(self.indices.keys())}`"
            )

        # Precompute X keys for faster access
        # We unpack keys like X_s1_a1->arr into a dict cache[1][1]->arr
        self._fast_key_access = self._str_keys_to_nested_dict(self._data.keys())
        self._is_npz = None

    @classmethod
    def from_npz(cls, path, lazy=True, close_filehandle=False):
        """
        Load from a single NPZ file. If lazy=True, this can be very slow
        for large amounts of arrays.

        Parameters
        ----------
        path : str
            Path to the NPZ file
        lazy : bool, optional=True
            Whether to let Numpy load arrays on demand, upon access (True)
            or preload everything in memory (False)
        close_filehandle : bool, optional=False
            Whether to close the NPZ filehandle after reading some metadata. This will
            enable parallelism without preloading everything, but each access will suffer
            the overhead of opening the NPZ file again!

        Note
        ----
        NPZ files cannot be read in parallel (you'll see CRC32 errors and others). If you want
        to use ``DataLoader(..., num_workers=2)`` or above, you'll need to:

        - A) preload everything with ``lazy=False``. This will use more RAM and incur
          an initial waiting time.
        - B) use ``close_filehandle=True``. This will incur a penalty upon each access,
          because the NPZ file needs to be reloaded each time.

        """
        data = np.load(path)
        if not lazy:
            name = Path(path).stem
            data = dict(tqdm(data.items(), desc=f"Loading {name}"))
        if "idx_train" in data:
            indices = {
                key[4:]: data[key] for key in ["idx_train", "idx_test", "idx_val"] if key in data
            }
        else:
            indices = {"train": True}

        inst = cls(data, indices=indices)
        inst._path = path
        if close_filehandle:
            inst._data.close()
        return inst

    def _getitem_multi_X(self, accessor):
        """
        Note: This method might scale poorly and can end up being a bottleneck!
        Most of the time is spent accessing the NPZ file on disk, though.

        Some timings:

        >>> ds = MultiXTorchDataset.from_npz("ChEMBLDatasetProvider.npz")
        >>> %timeit _ = ds[0:2]
        2.91 ms ± 222 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        >>> %timeit _ = ds[0:4]
        5.59 ms ± 253 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        >>> %timeit _ = ds[0:8]
        11.4 ms ± 1.02 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
        >>> %timeit _ = ds[0:16]
        22.7 ms ± 1.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit _ = ds[0:32]
        44.7 ms ± 4.47 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit _ = ds[0:64]
        87 ms ± 2.74 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit _ = ds[0:128]
        171 ms ± 2.68 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """
        indices, single_item = _accessor_to_indices(accessor, full_size=len(self))

        if hasattr(self._data, "zip") and self._data.zip is None:
            # data was loaded from NPZ but fh was closed; we need to reopen
            data = np.load(self._path)
            must_close = True
        else:
            data = self._data
            must_close = False

        result_X = []
        for index in indices:
            X_subresult = []
            for key in self._fast_key_access[index]:
                X_subresult.append(torch.tensor(data[key]))
            result_X.append(X_subresult)

        if must_close:
            data.close()

        if single_item:
            return result_X[0], self.data_y[indices]
        return result_X, self.data_y[indices]

    def _getitem_single_X(self, index):
        return self.data_X[index], self.data_y[index]

    def __getitem__(self, index):
        if self.data_X is None:
            return self._getitem_multi_X(index)
        return self._getitem_single_X(index)

    def _shape_X(self):
        if self.is_single_X():
            return torch.Size(self.data_X.shape)

        keys = [self._key_to_ints(k) for k in self._data.keys() if k.startswith("X")]
        shape = []
        for dim in range(len(keys[0])):
            shape.append(len(set([k[dim] for k in keys])))
        return torch.Size(tuple(shape))

    def is_single_X(self):
        X_keys = [k for k in self._data.keys() if k.startswith("X")]
        return len(X_keys) == 1 and X_keys[0] == "X"

    def _str_keys_to_nested_dict(self, keys):
        X_keys = [(k, self._key_to_ints(k)) for k in keys if k.startswith("X")]
        X_keys.sort(key=lambda x: x[1])
        result = defaultdict(list)
        for k, ints in X_keys:
            result[ints[0]].append(k)
        return result

    @staticmethod
    def _key_to_ints(key: str) -> List[int]:
        """
        NPZ keys are formatted with this syntax:

        ``{X|y}_{1-character str}{int}_{1-character str}{int}_``

        We split by underscores and extract the ints into a list
        """
        # key[2:] removes the X_ or y_ prefix
        # key.rstrip("_") removes the trailing underscore
        prefixed_numbers = key[2:].rstrip("_").split("_")
        numbers = []
        for field in prefixed_numbers:
            numbers.append(int(field[1:]))  # [1:] removes the pre-int prefix
        return numbers

    def __len__(self):
        return self.data_y.shape[0]


class AwkwardArrayDataset(_NativeTorchDataset):
    """
    Loads an Awkward array of Records.

    The structure of the array dimensions needs to be:

    - List of systems

    ---- X1
    ---- X2
    ---- ...
    ---- Xn
    ---- y

    However, X1...Xn, y are accessed by positional index, as a string.

    So, to get all the X1 vectors for all systems, you'd do:

    X1 = data["0"]
    X2 = data["1"]

    Since ``y`` is always the last one you can use the ``data.fields``
    list:

    y = data[data.fields[-1]]

    This is essentially what ``__getitem__`` is doing for you.

    It will try to consolidate tensors whenever possible, as long as
    they have the same shape. If they do not, then you'll get a list
    of tensors instead.

    If this is the case, make sure to provide a suitable ``collate_fn``
    function for the corresponding Dataloader! More info:

    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn

    Notes
    -----
    With several tensors per system, but all of the same shape, it is faster:

    >>> awk = AwkwardArrayDataset.from_parquet("same_shape.parquet")
    >>> %timeit _ = awk[:50]
    2.38 ms ± 286 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> awk = AwkwardArrayDataset.from_parquet("different_shape.parquet")
    >>> %timeit _ = awk[:50]
    9.32 ms ± 252 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    This is probably due to the Awkward->Numpy->Torch conversions that need
    to happen for each different-shape sub-tensor. Look in ``__getitem__``
    for bottlenecks.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = []
        fields = self.data.fields
        for f in fields[:-1]:
            tensors = self.data[index, f]
            try:
                tensors = torch.from_numpy(ak.to_numpy(tensors))
            except ValueError:
                # This can be slow with a lot of tensors (index > 1000?)
                tensors = [torch.from_numpy(ak.to_numpy(t)) for t in tensors]
            X.append(tensors)
        y = torch.tensor(self.data[index, fields[-1]])
        return X, y

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    @classmethod
    def from_parquet(cls, path, **kwargs):
        return cls(ak.from_parquet(path, **kwargs))


def _accessor_to_indices(accessor, full_size):
    single_item = False
    if accessor is True:
        indices = range(full_size)
    elif accessor is False:
        return tuple([], [])
    else:
        try:
            indices_arr = np.asarray(accessor)
            if len(indices_arr.shape) == 1:
                indices = indices_arr.tolist()
        except ValueError:
            pass
        if isinstance(accessor, (list, tuple)):
            if isinstance(accessor[0], int):
                indices = accessor
            elif isinstance(accessor[0], bool):
                indices = [i for i, value in enumerate(accessor) if value]
        elif isinstance(accessor, slice):
            indices = range(accessor.start or 0, accessor.stop or full_size, accessor.step or 1)
        elif isinstance(accessor, int):
            indices = [accessor]
            single_item = True

        return indices, single_item
