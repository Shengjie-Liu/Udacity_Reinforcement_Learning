# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# define model (actor and critic)

class actor(nn.Module):

    def __init__(self, state_size, action_size, fc1_units = 1028, fc2_units = 256, seed = 12):

        super(actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        # three fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # due to action belongs to [-1, 1]

        return x


class critic(nn.Module):

    def __init__(self, state_size, action_size, fc1_units = 1028, fc2_units = 256, seed = 13):

        super(critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        # three fully connected layers
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):

        inp = torch.cat((state, action), dim = 1)
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # due to action value belongs to R

        return x
