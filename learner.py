import torch
import torch.optim as optim
import numpy as np

from time import sleep
import os

from model import QNet
from replay_memory import NStepMemory, ReplayMemory


def learner_process(n_actors):
    leaner = Learner(n_actors)
    leaner.run()


class Learner:
    def __init__(self, n_actors, device='cuda:0'):
        # params
        self.gamma = 0.99
        self.alpha = 0.6
        self.bootstrap_steps = 1
        self.initial_exploration = 50000
        self.device = device
        self.n_epochs = 0
        self.n_actors = n_actors
        
        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')
        self.net_path = os.path.join(
            './', 'logs', 'model', 'net.pt')
        self.target_net_path = os.path.join(
            './', 'logs', 'model', 'target_net.pt')
        
        # memory
        self.memory_size = 500000
        self.batch_size = 128
        self.memory_load_interval = 10
        self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)

        # net
        self.net_save_interval = 50
        self.target_update_interval = 1000
        self.net = QNet(self.net_path, self.device).to(self.device)
        self.target_net = QNet(self.target_net_path, self.device).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.net.save()
        self.target_net.save()
        self.optim = optim.RMSprop(self.net.parameters(), lr=0.00025/4.0, alpha=0.95, eps=1.5e-7, centered=True)
    
    def run(self):
        while True:
            if self.replay_memory.size > self.initial_exploration:
                self.train()
            self.interval()
    
    def train(self):
        #print('1')
        batch, index, weights = self.replay_memory.sample(self.device)
        #print('2', batch['state'].size())

        # q_value
        q_value = self.net(batch['state'])
        #print('2.5')
        q_value = q_value.gather(1, batch['action'])

        #print('3')
        # target q_value
        with torch.no_grad():
            next_action = torch.argmax(
                self.net(batch["next_state"]), 1).view(-1, 1)
            next_q_value = self.target_net(
                    batch["next_state"]).gather(1, next_action)
            #print(batch['done'].view(-1))
            target_q_value = batch["reward"] + (self.gamma**self.bootstrap_steps) * next_q_value * (1 - batch['done'])
        
        # update
        self.optim.zero_grad()
        loss = torch.mean(0.5 * (q_value - target_q_value) ** 2)
        loss.backward()
        self.optim.step()

    def interval(self):
        self.n_epochs += 1
        if self.n_epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        if self.n_epochs % self.net_save_interval == 0:
            self.net.save()
            self.target_net.save()
        if self.n_epochs % self.memory_load_interval == 0:
            for i in range(self.n_actors):
                self.replay_memory.load(self.memory_path, i)
