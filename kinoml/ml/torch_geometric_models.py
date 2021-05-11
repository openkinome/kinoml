"""
Implementation of some Deep Neural Networks in Pytorch using Pytorch Geometric.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class _BaseModule(nn.Module):
    needs_input_shape = True

    @staticmethod
    def estimate_input_shape(input_sample):
        # Take the first batch [0]
        # From (adjacency, per node) tensors, take per node [1]
        # From per node tensor, take the number of features [1]
        return input_sample[0][1][1].shape


class GraphConvolutionNeuralNetwork(_BaseModule):
    """
    Builds a Graph Convolutional Network and a feed-forward pass

    Parameters
    ----------
    nb_nodes_features : int, default=9
        Number of features per node in the graph.
    embedding_shape : int, default=100
        Dimension of latent vector.
    output_shape : int, default=1
        Size of the last unit, representing delta_g_over_kt in our setting.
    _activation : torch function, default=relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self, input_shape, embedding_shape=100, output_shape=1, activation=F.relu
    ):
        super().__init__()
        self.input_shape = input_shape[0] # nb of per node features
        self.embedding_shape = embedding_shape
        self.output_shape = output_shape
        self._activation = activation

        self.GraphConvLayer1 = GCNConv(self.input_shape, self.embedding_shape)
        self.GraphConvLayer2 = GCNConv(self.embedding_shape, self.output_shape)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self._activation(self.GraphConvLayer1(x, edge_index))
        return self.GraphConvLayer2(x, edge_index)
