import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseModule(nn.Module):
    needs_input_shape = True

    @staticmethod
    def estimate_input_shape(input_sample):
        if type(input_sample) == list:
            # The shape of the ligand and the shape of the protein
            return list([input_sample[0][0].shape, input_sample[0][1].shape])
        else:
            return input_sample.shape[1:]


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
        self.input_shape = input_shape[0]
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
        x = x.float()
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


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
    input_shape : tuple
        Dimension of input tensor, with `nb_char`, expected number of possible characters and
        maximum length of SMILES.
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


    def __init__(
        self,
        input_shape,
        embedding_shape=300,
        kernel_shape=10,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.nb_char = self.input_shape[0]
        self.max_length = self.input_shape[1]
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


class ConvolutionNeuralNetworkRegressionKinaseInformed(_BaseModule):
    """
    Builds a Convolutional Neural Network and a feed-forward pass for a kinase-ligand setting.

    Parameters
    ----------
    input_shape : tuple
        Dimension of input tensors, for the ligand and the protein.
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


    def __init__(
        self,
        input_shape,
        embedding_shape=300,
        kernel_shape=10,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super().__init__()

        self.input_shape = input_shape
        # Ligand shape
        # nb_char = self.input_shape[0][0],
        # max_length_smiles = self.input_shape[0][1],

        # Protein shape
        # nb_residues = self.input_shape[1][0],
        # nb_amino_acids= self.input_shape[1][1],

        self.embedding_shape = embedding_shape
        self.kernel_shape = kernel_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        # Convolution on ligand
        self.convolution_ligand = nn.Conv1d(
            # in_channels=nb_char,
            in_channels = self.input_shape[0][0],
            out_channels=self.embedding_shape,
            kernel_size=self.kernel_shape,
        )
        self.temp_ligand = (self.input_shape[0][1] - self.kernel_shape + 1) * self.embedding_shape


        # Convolution on protein
        self.convolution_protein = nn.Conv1d(
            in_channels=self.input_shape[1][1],
            out_channels=self.embedding_shape,
            kernel_size=self.kernel_shape,
        )
        self.temp_protein = (self.input_shape[1][0] - self.kernel_shape + 1) * self.embedding_shape

        self.fully_connected_1 = nn.Linear(self.temp_ligand + self.temp_protein, self.hidden_shape)
        self.fully_connected_out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x_ligand, x_protein):
        """
        Defines the foward pass for given two inputs: ligand and protein.
        """
        x_lig = self._activation(self.convolution_ligand(x_ligand))
        x_prot = self._activation(self.convolution_protein(x_protein))

        x_lig = torch.flatten(x_lig, 1)
        x_prot = torch.flatten(x_prot, 1)

        x = torch.cat((x_lig, x_prot), dim=1)

        x = self._activation(self.fully_connected_1(x))

        return self.fully_connected_out(x)