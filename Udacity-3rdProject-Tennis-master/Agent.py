import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from _collections import deque, namedtuple
import random
import numpy as np
import copy
from model import actor, critic

""" Global variable """
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
ALPHA = 1e-2
GAMMA = 0.99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck Process to set action noise for exploration
    """
    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process """
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """ Reset the internal state (noise) to mean """
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a standard normal noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=len(x))
        self.state = x + dx
        return self.state


class replaybuffer:
    """
    define replaybuffer to store the experienced tuple (and decorrelate)
    """
    def __init__(self, seed, action_size):
        """
        initialize the replaybuffer
        :param seed: random seed for replaybuffer
        :param action_size: dimension of agent action
        """
        self.seed = random.seed(seed)
        self.action_size = action_size

        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", 'done'])

    def add(self, state, action, reward, next_state, done):
        """ add new sample into replaybuffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ randomly sample a batch of experiences from memory"""
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
        """ return the current size of internal replaybuffer"""
        return len(self.memory)


class agent:
    """ define the agent to interact with the environment"""
    def __init__(self, state_size, action_size, seed, lr_actor, lr_critic):
        """
        initilize agent by defining actor, critic and their targets
        :param state_size: dimension of agent state
        :param action_size: dimension of agent action
        :param seed: random seed for replaybuffer
        :param lr_actor: learning_rate of actor network
        :param lr_critic: learning_rate of critic network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.noise = OrnsteinUhlenbeck(action_size, mu=0., theta=0.15, sigma=0.05)
        self.actor = actor(state_size, action_size).to(device)
        self.actor_target = actor(state_size, action_size).to(device)
        self.critic = critic(state_size, action_size).to(device)
        self.critic_target = critic(state_size, action_size).to(device)
        self.copy(self.actor, self.actor_target)
        self.copy(self.critic, self.critic_target)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = lr_critic)
        self.memory = replaybuffer(seed, action_size)

    def reset(self):
        """ reset noise for every episode """
        self.noise.reset()

    def copy(self, local, target):
        """
        hardcopy the local network parameters to target network parameters at first episode
        :param local: local network
        :param target: target network
        """
        for local_params, target_params in zip(local.parameters(), target.parameters()):
            target_params.data.copy_(local_params.data)

    def collect(self, state, action, reward, next_state, done):
        """ store new sample into internal memory and learn the parameters"""
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, noise = True):
        """
        choose action based on the policy function approximator
        :param state: agent current state
        :param noise: exploration or not
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        """
        use ddpg algorithms to update parameters of local networks and use softupdate
        to update parameters of target networks
        :param experiences: experienced tuple samples
        """

        states, actions, rewards, next_states, dones = experiences
        TD_target = rewards + GAMMA*self.critic_target(next_states, self.actor_target(next_states))
        critic_loss = F.mse_loss(TD_target, self.critic(states, actions))
        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.optim_critic.step()

        value_max = -self.critic(states, self.actor(states)).mean()
        self.optim_actor.zero_grad()
        value_max.backward()
        self.optim_actor.step()
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local, target):
        """
        soft update for parameters of target networks
        :param local: local network
        :param target: target network
        """
        for local_params, target_params in zip(local.parameters(), target.parameters()):
            target_params.data.copy_(ALPHA*local_params.data + (1 - ALPHA)*target_params.data)

