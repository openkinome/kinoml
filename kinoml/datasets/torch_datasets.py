import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, systems, measurements):
        assert len(systems) == len(measurements), "Systems and Measurements must match in size!"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # note we are using as_tensor to _avoid_ copies if possible
        self.systems = systems
        self.measurements = measurements

    def __getitem__(self, index):
        # TODO: Since DataLoader will call this method to retrieve the batches,
        #       we could delay featurizations until this point! This will require
        #       some kind of computation graph (Dask? torch.Transforms?)
        # TODO: requires_grad=True _not_ required?
        X = torch.as_tensor(self.systems[index], device=self.device, dtype=torch.float)
        y = torch.as_tensor(self.measurements[index], device=self.device)
        return X, y

    def __len__(self):
        return len(self.systems)

    def as_dataloader(self, **kwargs):
        return DataLoader(dataset=self, **kwargs)
