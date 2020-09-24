from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from ..core.measurements import null_observation_model as _null_observation_model


class PrefeaturizedTorchDataset(Dataset):
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
        return DataLoader(dataset=self, **kwargs)

    def estimate_input_size(self):
        return self.systems[0].shape


class TorchDataset(PrefeaturizedTorchDataset):
    def __init__(
        self,
        systems,
        measurements,
        featurizer=None,
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


class XyNpzTorchDataset(Dataset):
    def __init__(self, npz):
        data = np.load(npz)
        self.data_X = torch.as_tensor(data["X"], dtype=torch.float32)
        self.data_y = torch.as_tensor(data["y"], dtype=torch.float32)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]

    def __len__(self):
        return self.data_X.shape[0]

    def input_size(self):
        return self.data_X.shape[1]

    def as_dataloader(self, train_test_split=None, **kwargs):
        if train_test_split is not None and (0.0 < train_test_split < 1.0):
            indices = list(range(len(self)))
            np.random.shuffle(indices)
            test_split = int(np.floor(train_test_split * len(self)))
            train_indices, test_indices = indices[:test_split], indices[test_split:]
            train = DataLoader(self, sampler=SubsetRandomSampler(train_indices), **kwargs)
            test = DataLoader(self, sampler=SubsetRandomSampler(test_indices), **kwargs)
            return train, test
        # else
        return DataLoader(self, **kwargs)
