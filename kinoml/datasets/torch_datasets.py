"""
Helper classes to convert between DatasetProvider objects and
Dataset-like objects native to the PyTorch ecosystem
"""
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import TorchDataset as _NativeTorchDataset, DataLoader as _DataLoader

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
        return self.data_X.shape[1]
