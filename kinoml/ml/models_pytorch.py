"""
Implementation in Pytorch of some Deep Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    """
    Builds a Pytorch model (a Dense Neural Network) and a feed-forward pass

    Parameters
    ==========
    input_dim : int, optional=512
        Expected shape of the input data

    Returns
    =======
    model : a feed-forward pass of the dense neural network with activation functions
    """
    def __init__(self, input_dim=512):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 350) # fc1: 1st fully connected layer with 350 nodes
        self.fc2 = nn.Linear(350, 200) # fc2: 2nd fully connected layer with 200 nodes
        self.dropout1 = nn.Dropout(0.2) # dropout1: 1st dropout layer
        self.fc3 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        x = F.relu(self.fc1(x)) # All activations are relu expect for the last layer which is a sigmoid
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.sigmoid(self.fc6(x))


class CNN(nn.Module):
    """
    Builds a Pytorch model (a Convolutional Neural Network) and a feed-forward pass

    Parameters
    ==========
    input_shape : tuple of int
        Expected shape of the input data

    Returns
    =======
    model : a feed-forward pass of the convolutional neural network with activation functions
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=53, out_channels=100, kernel_size=10) # conv : 1D convolution
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(22*100, 64)
        #self.BN1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        #self.BN2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        #print('Input shape before conv : ' , x.size())
        x = self.conv(x)
        x = F.relu(x)
        #print('Input shape after conv : ' , x.size())
        x = F.max_pool1d(x, 4)
        #print('Input shape after 4 maxpool : ' , x.size())
        x = torch.flatten(x, 1)
        #print('Input shape after flatten : ' , x.size())
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class GCN(nn.Module):
    pass

