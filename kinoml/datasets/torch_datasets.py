from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, systems, measurements):
        assert len(systems) == len(measurements), "Systems and Measurements must match in size!"
        self.systems = systems
        self.measurements = measurements

    def __getitem__(self, index):
        return self.systems[index], self.measurements[index]

    def __len__(self):
        return len(self.systems)
