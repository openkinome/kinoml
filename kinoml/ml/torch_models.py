import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseModule(nn.Module):
    @staticmethod
    def estimate_input_shape(input_sample):
        return input_sample.shape[1]


class NeuralNetworkRegression(_BaseModule):
    """
    Builds a Pytorch model (a vanilla neural network) and a feed-forward pass.

    Parameters
    ----------
    input_shape : int
        Dimension of the input vector.
    hidden_shape : int, default=100
        Number of units in the hidden layer.
    output_shape : int, default=1
        Size of the last unit, representing delta_g_over_kt in our setting.
    _activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(self, input_shape, hidden_shape=100, output_shape=1, activation=F.relu):
        super().__init__()

        self._activation = activation
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        # Fully connected layer
        self.fully_connected_1 = nn.Linear(self.input_shape, self.hidden_shape)
        # Output
        self.fully_connected_out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'.
        """
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


class DenseNeuralNetworkRegression(_BaseModule):
    """
    Builds a Dense Neural Network and a feed-forward pass.

    Parameters:
        # TODO
    """

    def __init__(self, input_shape: int):
        super().__init__()
        self.input_shape = input_shape
        # fc1: 1st fully connected layer with 350 nodes
        self.fc1 = nn.Linear(self.input_shape, 350)
        # fc2: 2nd fully connected layer with 200 nodes
        self.fc2 = nn.Linear(350, 200)
        # dropout1: 1st dropout layer
        self.dropout1 = nn.Dropout(0.2)
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


class ConvolutionNeuralNetworkRegression(_BaseModule):
    """
    Builds a Convolutional Neural Network and a feed-forward pass.

    Parameters
    ----------
    nb_char : int, default=53
        Expected number of possible characters
        For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    embedding_shape : int, default=200
        Dimension of the embedding after convolution.
    kernel_shape : int, default=10
        Size of the kernel for the convolution.
    hidden_shape : int, default=100
        Number of units in the hidden layer.
    output_shape : int, default=1
        Size of the last unit, representing delta_g_over_kt in our setting.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        embedding_shape=300,
        kernel_shape=10,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super(ConvolutionNeuralNetworkRegression, self).__init__()

        self.nb_char = nb_char
        self.max_length = max_length
        self.embedding_shape = embedding_shape
        self.kernel_shape = kernel_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.convolution = nn.Conv1d(
            in_channels=self.nb_char, out_channels=self.embedding_shape, kernel_size=self.kernel_shape
        )
        self.temp = (self.max_length - self.kernel_shape + 1) * self.embedding_shape
        self.fully_connected_1 = nn.Linear(self.temp, self.hidden_shape)
        self.fully_connected_out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        x = self._activation(self.convolution(x))
        x = torch.flatten(x, 1)
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)
