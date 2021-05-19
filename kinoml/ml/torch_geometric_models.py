"""
Implementation of some Deep Neural Networks in Pytorch using Pytorch Geometric.
"""

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .torch_models import _BaseModule


class GraphConvolutionNeuralNetwork(_BaseModule):
    """
    Builds a Graph Convolutional Network and a feed-forward pass

    Parameters
    ----------
    nb_nodes_features : int
        Number of features per node in the graph.
    embedding_shape : int, default=100
        Dimension of latent vector.
    output_shape : int, default=1
        Size of the last unit, representing delta_g_over_kt in our setting.
    _activation : torch function, default=relu
        The activation function used in the hidden (only!) layer of the network.
    """

    needs_input_shape = True

    @staticmethod
    def estimate_input_shape(input_sample):
        # Take the first batch [0]
        return input_sample[0].num_node_features

    def __init__(self, input_shape, embedding_shape=100, output_shape=1, activation=F.relu):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape
        self.output_shape = output_shape
        self._activation = activation

        self.GraphConvLayer1 = GCNConv(self.input_shape, self.embedding_shape)
        self.GraphConvLayer2 = GCNConv(self.embedding_shape, self.output_shape)

    def forward(self, data):
        data = data[0]  # get the first one only?
        x, edge_index = data.x.float(), data.edge_index.long()
        x = self._activation(self.GraphConvLayer1(x, edge_index))
        return self.GraphConvLayer2(x, edge_index)
