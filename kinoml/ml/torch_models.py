import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseModule(nn.Module):
    needs_input_shape = True

    @staticmethod
    def estimate_input_shape(input_sample):
        """
        This static method takes the same input
        as ``.forward()`` would and estimates the
        incoming shape so the layers can be initialized
        properly.

        Most of the time, ``input_sample`` would be a
        Tensor, in which the first dimension corresponds
        to systems, and the second is the input shape
        we need.

        If your ``.forward()`` method takes something else
        than a Tensor, please adapt this method accordingly.
        """
        if type(input_sample) == list:
            return input_sample[0].shape[1]
        else: input_sample.shape[1]


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
        x = x[0]
        x = x.float()
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


class ListOfTupleNeuralNetworkregression(NeuralNetworkRegression):
    """
    This example model does not take a Tensor in, but a
    tuple of tensors. Each tensor has shape
    (n_systems, n_features).

    As a result, one needs to concatenate the results
    before passing it to the parent ``.forward()`` method.
    """

    @staticmethod
    def estimate_input_shape(input_sample):
        return sum(x.shape[1] for x in input_sample)

    def forward(self, x):
        return super().forward(torch.cat(x, dim=1))


class DenseNeuralNetworkRegression(_BaseModule):
    """
    Builds a Dense Neural Network and a feed-forward pass.

    Parameters
    ----------
    input_shape : int
        Dimension of the input vector.
    hidden_shape : list
        Number of units in each of the hidden layers.
    output_shape : int, default=1
        Size of the last unit, representing delta_g_over_kt in our setting.
    dropout_percentage : float
        The percentage of hidden to by dropped at random.
    _activation : torch function, default=relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        input_shape,
        hidden_shape=(350, 200, 100, 50, 16),
        output_shape=1,
        dropout_percentage=0.4,
        activation=F.relu,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.dropout_percentage = dropout_percentage
        self._activation = activation

        self.fully_connected_1 = nn.Linear(self.input_shape, self.hidden_shape[0])
        self.fully_connected_2 = nn.Linear(self.hidden_shape[0], self.hidden_shape[1])
        self.fully_connected_3 = nn.Linear(self.hidden_shape[1], self.hidden_shape[2])
        self.fully_connected_4 = nn.Linear(self.hidden_shape[2], self.hidden_shape[3])
        self.fully_connected_5 = nn.Linear(self.hidden_shape[3], self.hidden_shape[4])
        self.fully_connected_out = nn.Linear(self.hidden_shape[4], self.output_shape)

        self.dropout = nn.Dropout(self.dropout_percentage)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        x = self._activation(self.fully_connected_1(x))
        x = self._activation(self.fully_connected_2(x))
        x = self.dropout(x)
        x = self._activation(self.fully_connected_3(x))
        x = self._activation(self.fully_connected_4(x))
        x = self.dropout(x)
        x = self._activation(self.fully_connected_5(x))
        return self.fully_connected_out(x)


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
    activation : torch function, default=relu
        The activation function used in the hidden (only!) layer of the network.
    """

    needs_input_shape = False

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
        super().__init__()

        self.nb_char = nb_char
        self.max_length = max_length
        self.embedding_shape = embedding_shape
        self.kernel_shape = kernel_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.convolution = nn.Conv1d(
            in_channels=self.nb_char,
            out_channels=self.embedding_shape,
            kernel_size=self.kernel_shape,
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
