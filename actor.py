import torch
import numpy as np

import gym
import os

from model import QNet
from replay_memory import NStepMemory, ReplayMemory


def actor_process(actor_id, n_actors, device='cpu'):
    actor = Actor(actor_id, n_actors, device)
    actor.run()


class Actor:
    def __init__(self, actor_id, n_actors, device='cpu'):
        # params
        self.gamma = 0.95
        self.epsilon = 0.4 ** (1 + actor_id * 7 / (n_actors - 1))
        self.bootstrap_steps = 1
        self.alpha = 0.6
        self.device = device
        self.actor_id = actor_id

        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')
        self.net_path = os.path.join(
            './', 'logs', 'model', 'net.pt')
        self.target_net_path = os.path.join(
            './', 'logs', 'model', 'target_net.pt')

        # memory
        self.memory_size = 10000
        self.batch_size = 32
        self.stack_count = 4
        self.memory_save_interval = 10
        self.n_steps_memory = NStepMemory(self.bootstrap_steps, self.gamma)
        self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)

        # net
        self.net_load_interval = 5
        self.net = QNet(self.net_path)
        self.target_net = QNet(self.target_net_path)
        self.target_net.load_state_dict(self.net.state_dict())

        # env
        self.env = gym.make('CartPole-v0')
        self.episode_reward = 0
        self.n_episodes = 0
        self.n_steps = 0
        self.state = self.env.reset()
    
    def run(self):
        while True:
            self.step()

    def step(self):
        state = self.state
        action = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        if self.n_steps > 195:
            reward = 0
        elif done:
            reward = -1
        else:
            reward = 0
        self.episode_reward = reward
        self.n_steps += 1

        self.n_steps_memory.add(state, action, reward, self.stack_count)
        if self.stack_count > 1:
            self.stack_count -= 1
        
        if self.n_steps > self.bootstrap_steps:
            state, action, reward, stack_count = self.n_steps_memory.get()
            self.replay_memory.add(state, action, reward, done, stack_count)
        self.state = next_state.copy()

        if done:
            self.reset()
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(2)
        else:
            with torch.no_grad():
                q_val = self.net(state)
                action = q_val.argmax().item()
        return action
    
    def reset(self):
        if self.n_episodes % 50 == 0:
            print('episodes:', self.n_episodes, 'actor_id:', self.actor_id, 'return:', self.n_steps)

        self.state = self.env.reset()
        self.episode_reward = 0
        self.n_episodes += 1
        self.n_steps = 0
        self.stack_count = 4

        # reset n_step memory
        self.n_steps_memory = NStepMemory(self.bootstrap_steps, self.gamma)

        # save replay memory
        if self.n_episodes % self.memory_save_interval == 0:
            self.replay_memory.save(self.memory_path, self.actor_id)
            self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)
        
        # load net
        if self.n_episodes % self.net_load_interval == 0:
            self.net.load()
            self.target_net.load()