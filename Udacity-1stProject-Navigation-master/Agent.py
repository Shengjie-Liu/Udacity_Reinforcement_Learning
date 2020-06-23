# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np
from model import Qnetwork

# define global variables
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
EPSILON = 0.05 # epsilon_greedy policy
ALPHA = 1e-3 # soft_update rate
LR = 5e-4 # learning rate
UPDATE_EVERY = 4 # update parameter for fixed Q target every ...
GAMMA = 0.99 # Discount factor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define agent for the RL task

class Agent:

    def __init__(self, seed, state_size, action_size, net_type="dqn"):
        """if net_type is dqn, perform deep Q network; if ddqn, perform double deep Q network"""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.net_type = net_type
        # replay buffer
        self.memory = replaybuffer(action_size, BATCH_SIZE, seed)
        # define target and local Q network
        self.qnetwork_local = Qnetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = Qnetwork(state_size, action_size, seed).to(device)
        # define optimizer for qnetwork_local
        self.optim = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # define time step for soft updating cycle
        self.time_step = 0

    def collect(self, state, action, reward, next_state, done):
        # collect the new sample
        self.memory.add(state, action, reward, next_state, done)
        # use time step to decide if it needs to learn or not
        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, ALPHA)

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        # get action_values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_vals = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # use epsilon_greedy policies to decide which action to take
        policy = np.ones(self.action_size) * (EPSILON / self.action_size)
        best = torch.argmax(action_vals).item()
        policy[best] = 1 - EPSILON + (EPSILON / self.action_size)
        return np.random.choice(np.arange(self.action_size), p=policy)

    def learn(self, experiences, alpha):

        states, actions, rewards, next_states, dones = experiences
        # parameter learning for local network
        if self.net_type == "dqn":
            TD_target = rewards + GAMMA * (self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)) * (
                        1 - dones)
        if self.net_type == "ddqn":
            best = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            TD_target = rewards + GAMMA * (self.qnetwork_target(next_states).detach().gather(1, best))
        TD_estimate = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(TD_target, TD_estimate)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # parameter soft updating for target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, alpha)

    def soft_update(self, local_network, target_network, alpha):

        for local_params, target_params in zip(local_network.parameters(), target_network.parameters()):
            target_params.data.copy_(alpha * local_params.data + (1 - alpha) * target_params.data)


# define replay buffer

class replaybuffer:

    def __init__(self, action_size, batch_size, seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)