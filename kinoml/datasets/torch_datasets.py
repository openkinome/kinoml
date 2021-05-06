"""
Helper classes to convert between DatasetProvider objects and
Dataset-like objects native to the PyTorch ecosystem
"""
from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset as _NativeTorchDataset, DataLoader as _DataLoader

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
        self, systems, measurements, observation_model: callable = _null_observation_model
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

    @lru_cache(maxsize=100_000)
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
            self.measurements[index], device=self.device, requires_grad=True, dtype=torch.float
        )
        return X, y


class XyNpzTorchDataset(_NativeTorchDataset):
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

    def __init__(self, npz):
        data = np.load(npz)
        self.data_X = torch.as_tensor(data["X"])
        self.data_y = torch.as_tensor(data["y"])
        if "idx_train" in data:
            self.indices = {
                key[4:]: data[key] for key in ["idx_train", "idx_test", "idx_val"] if key in data
            }
        else:
            self.indices = {"train": True}

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]

    def __len__(self):
        return self.data_X.shape[0]

    def input_size(self):
        # Optional for some models!
        return self.data_X.shape[1]


class MultiXNpzTorchDataset(_NativeTorchDataset):
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
    npz : str
        Path to the NPZ file.

    Notes
    -----
    This object is better paired with the output of ``DatasetProvider.to_dict_of_arrays``.
    """

    def __init__(self, npz):
        self.data = data = np.load(npz)
        self.data_y = torch.tensor(data["y"])
        if self.is_single_X():
            self.data_X = torch.tensor(data["X"])
        else:
            self.data_X = None

        self.shape_X = self._shape_X()
        self.shape_y = self.data_y.shape

        if "idx_train" in data:
            self.indices = {
                key[4:]: data[key] for key in ["idx_train", "idx_test", "idx_val"] if key in data
            }
        else:
            self.indices = {"train": True}

    def _getitem_multi_X(self, accessor):
        single_item = False
        if accessor is True:
            indices = list(range(self.shape_y[0]))
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
                indices = range(*accessor)
            elif isinstance(accessor, int):
                indices = [accessor]
                single_item = True

        result_X = []
        for index in indices:
            X_prefix = f"X_s{index}"
            X_subresult = []
            this_system_keys = [k for k in self.data.keys() if k.startswith(X_prefix)]
            for key in sorted(this_system_keys, key=self._key_to_ints):
                X_subresult.append(torch.tensor(self.data[key]))
            result_X.append(X_subresult)

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

        keys = [self._key_to_ints(k) for k in self.data.keys() if k.startswith("X")]
        shape = []
        for dim in range(len(keys[0])):
            shape.append(len(set([k[dim] for k in keys])))
        return torch.Size(tuple(shape))

    def is_single_X(self):
        X_keys = [k for k in self.data.keys() if k.startswith("X")]
        return len(X_keys) == 1 and X_keys[0] == "X"

    @staticmethod
    def _key_to_ints(key: str) -> List[int]:
        """
        NPZ keys are formatted with this syntax:

        ``{X|y}_{1-character str}{int}_{1-character str}{int}``

        We split by underscores and extract the ints into a list
        """
        prefixed_numbers = key[2:].split("_")  # [2:] removes the X_ or y_ prefix
        numbers = []
        for field in prefixed_numbers:
            numbers.append(int(field[1:]))  # [1:] removes the pre-int prefix
        return numbers

    def __len__(self):
        return self.data["y"].shape[0]