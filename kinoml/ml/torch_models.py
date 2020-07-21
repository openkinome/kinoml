import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkRegression(nn.Module):
    """
    Builds a Pytorch model (a Dense Neural Network) and a feed-forward pass
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
        Defines the foward pass for a given input 'x'
        """
        x = self._activation(self.fully_connected_1(x))  # Activations are ReLU
        return self.fully_connected_out(x)
