"""
Implementation of some Deep Neural Networks in Pytorch using Pytorch Geometric.
"""

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .torch_models import _BaseModule


class GraphConvolutionNeuralNetwork(_BaseModule):
    """
    Builds a Graph Convolutional Network and a feed-forward pass

    Parameters
    ----------
    input_shape : int
        Number of features per node in the graph.
    embedding_shape : int, default=100
        Dimension of latent vector.
    hidden_shape : int, default=50
        Dimension of the hidden shape.
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

    def __init__(self, input_shape, embedding_shape=100, hidden_shape=50, output_shape=1, activation=F.relu):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.GraphConvLayer1 = GCNConv(self.input_shape, self.embedding_shape)
        self.GraphConvLayer2 = GCNConv(self.embedding_shape, self.output_shape)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self._activation(self.GraphConvLayer1(x, edge_index))
        return self.GraphConvLayer2(x, edge_index)

