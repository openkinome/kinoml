import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(512, 350) # fc1: 1st fully connected layer with 350 nodes
        self.fc2 = nn.Linear(350, 200) # fc2: 2nd fully connected layer with 200 nodes
        self.dropout1 = nn.Dropout(0.2) # dropout1: 1st dropout layer
        self.fc3 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # All activations are relu expect for the last layer which is a sigmoid
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.sigmoid(self.fc6(x))