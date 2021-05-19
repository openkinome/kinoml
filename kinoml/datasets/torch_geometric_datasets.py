"""
Helper classes to convert between DatasetProvider objects and
Dataset-like objects native to the PyTorch ecosystem
"""

import awkward as ak
import torch
from torch.utils.data import Dataset as _NativeTorchDataset
from torch_geometric.data import Data


# Disable false positive lint with torch.tensor
# see https://github.com/pytorch/pytorch/issues/24807
# pylint: disable=not-callable


class AwkwardArrayGeometricDataset(_NativeTorchDataset):
    """
    Loads an Awkward array of Records suitable for PyTorch Geometric.
    It assumes the following:

    - The Awkward array contains three fields: 0, 1 and 2
    - 0: Conn. matrix  --> Data's ``edge_index``
    - 1: Node features --> Data's ``x``
    - 2: y labels

    If more attributes are needed, you need to modify ``__getitem__`` logic
    """

    def __init__(self, data):
        assert len(data.fields) == 3, (
            f"Graph datasets should only contain three groups: "
            "0, 1 and 2 (conn. matrix, node features, y; respectively)"
        )
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        fields = self.data.fields
        edge_index = self.data[index, fields[0]]
        node_features = self.data[index, fields[1]]
        y = torch.tensor(self.data[index, fields[2]])
        X = [
            Data(x=torch.tensor(nf), edge_index=torch.tensor(ei).long())
            for (nf, ei) in zip(node_features, edge_index)
        ]
        return X, y

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    @classmethod
    def from_parquet(cls, path, **kwargs):
        return cls(ak.from_parquet(path, **kwargs))
