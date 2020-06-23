# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random


# define Qnetwork as function approximator for mapping from state to different action value functions

class Qnetwork(nn.Module):

    """
    three fully connected layers with relu activation and one output layer
    """

    def __init__(self, state_size, action_size, seed, fc1_units=1024, fc2_units=256, fc3_units=128):
        super(Qnetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)