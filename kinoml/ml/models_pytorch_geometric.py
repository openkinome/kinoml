"""
Implementation of some Deep Neural Networks in Pytorch using Pytorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Builds a Pytorch model (a Graph Convolutional Network) and a feed-forward pass

    Parameters
    ==========
    nb_nodes_features : int, optional=3
        Number of features per node in the graph.

    Returns
    =======
    model : a feed-forward pass of the GCN with activation functions
    """
    pass

    def __init__(self, nb_nodes_features):
        super(GCN, self).__init__()
        self.nb_nodes_features = nb_nodes_features
        self.GCLayer1 = GCNConv(self.nb_nodes_features, 64)
        self.GCLayer2 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.GCLayer1(x, edge_index))
        x = self.GCLayer2(x, edge_index)
        return torch.sigmoid(x)


