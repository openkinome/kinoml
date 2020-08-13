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


class DNN(nn.Module):
    """
    Builds a Dense Neural Network and a feed-forward pass

    Parameters:
        input_dim: Expected shape of the input data
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


class CNN(nn.Module):
    """
    Builds a Convolutional Neural Network and a feed-forward pass

    Parameters:
        input_shape: Expected shape of the input data
    """

    def __init__(self, input_shape: int = 53):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.conv = nn.Conv1d(in_channels=self.input_shape, out_channels=100, kernel_size=10)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(22 * 100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 4)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
