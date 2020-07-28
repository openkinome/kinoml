from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader

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
        X = torch.as_tensor(self.systems[index], device=self.device, dtype=torch.float)
        y = torch.as_tensor(self.measurements[index], device=self.device)
        return X, y

    def __len__(self):
        return len(self.systems)

    def as_dataloader(self, **kwargs):
        return DataLoader(dataset=self, **kwargs)

    def estimate_input_size(self):
        return self.systems[0].shape


class TorchDataset(PrefeaturizedTorchDataset):
    def __init__(self, featurizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        X = torch.as_tensor(
            self.featurizer(self.systems[index]).featurizations[self.featurizer.name],
            device=self.device,
            dtype=torch.float,
        )
        y = torch.as_tensor(self.measurements[index], device=self.device)
        return X, y
