import torch
import torch.nn as nn
import torch.nn.functional as F

class actor(nn.Module):
    """
    map current states to action i.e policy function approximator
    """
    def __init__(self, state_size, action_size, fc1_units = 256, fc2_units = 128):
        """
        initialize the policy network
        :param state_size: dimension of state
        :param action_size: dimension of action
        :param fc1_units: hidden units for first layer
        :param fc2_units: hidden units for second layer
        """
        super(actor,self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.b1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.b2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        policy network forward propagation
        :param state: agent current state
        :return: action (belongs to [-1, 1], use tanh activation)
        """
        x = self.b1(F.relu(self.fc1(state)))
        x = self.b2(F.relu(self.fc2(x)))
        return F.tanh(self.fc3(x))


class critic(nn.Module):
    """
    map current state and action to action value i.e. q-value
    """
    def __init__(self, state_size, action_size, fc1_units = 256, fc2_units = 128):
        """
        initialize the critic network
        :param state_size: dimension of state
        :param action_size: dimension of action
        :param fc1_units: hidden units for first layer
        :param fc2_units: hidden units for second layer
        """
        super(critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.b1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """
        critic network forward propagation
        :param state: dimension of state
        :param action: dimension of action
        :return: q-value (belongs to real number, no activation)
        """
        x = self.b1(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x, action), dim = 1)))
        return self.fc3(x)