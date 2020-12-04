import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkRegression(nn.Module):
    """
    Builds a Pytorch model (a vanilla neural network) and a feed-forward pass.

    Parameters
    ----------
        input_size : int, default=1024
                Dimension of the input vector.
        hidden_size : int, default=100
                Number of units in the hidden layer.
        output_size : int, default=1
                Size of the last unit, representing delta_g_over_kt in our setting.
        _activation : torch function, default: relu
                The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(self, input_size=1024, hidden_size=100, output_size=1, activation=F.relu):
        super(NeuralNetworkRegression, self).__init__()

        self._activation = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Fully connected layer
        self.fully_connected_1 = nn.Linear(self.input_size, self.hidden_size)
        # Output
        self.fully_connected_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'.
        """
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


class DenseNeuralNetworkRegression(nn.Module):
    """
    Builds a Dense Neural Network and a feed-forward pass.

    Parameters:
        # TODO
    """

    def __init__(self, input_dim: int = 512):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 350)  # fc1: 1st fully connected layer with 350 nodes
        self.fc2 = nn.Linear(350, 200)  # fc2: 2nd fully connected layer with 200 nodes
        self.dropout1 = nn.Dropout(0.2)  # dropout1: 1st dropout layer
        self.fc3 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        # All activations are relu expect for the last layer which is a sigmoid
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return torch.sigmoid(x)


class ConvolutionNeuralNetworkRegression(nn.Module):
    """
    Builds a Convolutional Neural Network and a feed-forward pass.

    Parameters
    ----------
        nb_char : int, default=53
                Expected number of possible characters
                For SMILES characters, we assume 53.
        max_length : int, default=256
                Maximum length of SMILES, set to 256.
        embedding_dim : int, default=200
                Dimension of the embedding after convolution.
        kernel_size : int, default=10
                Size of the kernel for the convolution.
        hidden_size : int, default=100
                Number of units in the hidden layer.
        output_size : int, default=1
                Size of the last unit, representing delta_g_over_kt in our setting.
        _activation : torch function, default: relu
                The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(self, nb_char=53, max_length=256, embedding_dim=300, kernel_size=10, hidden_size=100, output_size=1, activation=F.relu):
        super(ConvolutionNeuralNetworkRegression, self).__init__()

        self.nb_char = nb_char
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._activation = activation

        self.convolution = nn.Conv1d(in_channels=self.nb_char, out_channels=self.embedding_dim, kernel_size=self.kernel_size)
        self.temp = (self.max_length - self.kernel_size + 1)  * self.embedding_dim
        self.fully_connected_1 = nn.Linear(self.temp, self.hidden_size)
        self.fully_connected_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        x = self._activation(self.convolution(x))
        x = torch.flatten(x, 1)
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)
