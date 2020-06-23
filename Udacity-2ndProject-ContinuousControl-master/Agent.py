# Import libaries
from model import actor, critic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define global variables
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
ALPHA = 1e-2 # soft-update rate
UPDATE_EVERY = 20 # update parameter for every # step
GAMMA = 0.99 # discount factor
NOISE = 1e-3 # noise for action exploration
NUM_UPDATES = 10 # num of updates per update step


# define replaybuffer to decorrelate experienced tuples

class replaybuffer:

    def __init__(self, seed, action_size):
        self.seed = random.seed(seed)
        self.action_size = action_size

        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k = BATCH_SIZE)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# define agent including actor and critic

class agent:

    def __init__(self, state_size, action_size, seed, lr_actor, lr_critic):

        self.state_size = state_size
        self.action_size = action_size

        # define actor, critic and their targets
        self.actor = actor(state_size, action_size).to(device)
        self.actor_target = actor(state_size, action_size).to(device)
        self.critic = critic(state_size, action_size).to(device)
        self.critic_target = critic(state_size, action_size).to(device)
        self.copy(self.actor, self.actor_target)
        self.copy(self.critic, self.critic_target)

        # define optim
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = lr_critic)
        # define memory
        self.memory = replaybuffer(seed, action_size)
        # define time step (to decide when we update)
        self.time_step = 0

    def copy(self, local, target):
        for local_params, target_params in zip(local.parameters(), target.parameters()):
            target_params.data.copy_(local_params.data)

    def collect(self, state, action, reward, next_state, done):
        # store new sample
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1)%UPDATE_EVERY
        if len(self.memory) >= BATCH_SIZE:
            if self.time_step == 0:
                for _ in range(NUM_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, state):
        # choose action based on the policy function approximator
        state = torch.from_numpy(state).float().to(device)
        actions = self.actor(state)
        # this is for action_exploration
        noise = NOISE*torch.rand(actions.size()).to(device)
        return actions + noise

    def learn(self, experiences):
        # we will learn the parameters by using DDPG algorithm
        states, actions, rewards, next_states, dones = experiences
        # set TD target for value function
        TD_target = rewards + GAMMA*self.critic_target(next_states, self.actor_target(next_states))
        # for q-value (critic) function we use MSE loss
        critic_loss = F.mse_loss(TD_target, self.critic(states, actions))
        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.optim_critic.step()
        # for policy network we maximize the expected return i.e. Q-value
        # the negative here since we need to perform gradient ascent
        value_max = -self.critic(states, self.actor(states)).mean()
        self.optim_actor.zero_grad()
        value_max.backward()
        self.optim_actor.step()
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local, target):
        # do the soft update for the target network
        for local_params, target_params in zip(local.parameters(), target.parameters()):
            target_params.data.copy_(ALPHA*local_params.data + (1 - ALPHA)*target_params.data)
