import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from collections import deque
import os
from time import time, sleep
import gc
import fasteners
import pickle


class NStepMemory(dict):
    def __init__(self, memory_size=3, gamma=0.99):
        self.memory_size = memory_size
        self.gamma = gamma

        self.state = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.stack_count = deque(maxlen=memory_size)
    
    @property
    def size(self):
        return len(self.state)

    def add(self, state, action, reward, stack_count):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.stack_count.append(stack_count)

    def get(self):
        state = self.state.popleft()
        action = self.action.popleft()
        stack_count = self.stack_count.popleft()
        reward = sum([self.gamma ** i * r for i,r in enumerate(self.reward)])
        return state, action, reward, stack_count
    
    def clear(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
    
    def is_full(self):
        return len(self.state) == self.memory_size


class ReplayMemory:
    def __init__(self, memory_size=100000, batch_size=32, n_step=3, state_size=(84,84), action_repeat=4, n_stacks=4, alpha=0.4):#(3, 84, 84), alpha=0.4):
        self.index = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.state_size = (action_repeat,) + state_size
        self.action_repeat = action_repeat
        self.n_stacks = n_stacks // action_repeat
        self.alpha = alpha
        self.beta = 0.4
        self.beta_step = 0.00025 / 4

        self.memory = dict()
        self.memory['state'] = np.zeros((self.memory_size, *self.state_size), dtype=np.uint8)
        self.memory['action'] = np.zeros((self.memory_size, 1), dtype=np.int8)
        self.memory['reward'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['done'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['stack_count'] = np.zeros((self.memory_size,), dtype=np.int8)
        self.memory['priority'] = np.zeros((self.memory_size,), dtype=np.float32)

    @property
    def size(self):
        return min(self.index, self.memory_size)
    
    def add(self, state, action, reward, done, stack_count):
        index = self.index % self.memory_size
        self.memory['state'][index] = state * 255
        self.memory['action'][index] = action
        self.memory['reward'][index] = reward
        self.memory['done'][index] = 1 if done else 0
        self.memory['stack_count'][index] = stack_count
        self.index += 1
    
    def extend(self, memory):
        start_index = self.index % self.memory_size
        last_index = (start_index + memory['state'].shape[0]) % self.memory_size
        if start_index < last_index:
            index = [i for i in range(start_index, last_index)]
        else:
            index = [i for i in range(start_index, self.memory_size)] + [i for i in range(last_index)]
        index = np.array(index)
        
        for key in self.memory.keys():
            self.memory[key][index] = memory[key]

        self.index += memory['state'].shape[0]
    
    def fit(self):
        for key in self.memory.keys():
            self.memory[key] = self.memory[key][:self.size]
    
    def save(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    memory = torch.load(path, map_location=lambda storage, loc: strage)
                    self.extend(memory)
                    self.fit()
                    torch.save(self.memory, path)
                    lock.release()
                    gc.collect()
                    return
            else:
                if lock.acquire(blocking=False):
                    self.fit()
                    torch.save(self.memory, path)
                    lock.release()
                    gc.collect()
                    return
            sleep(np.random.random()+2)

    
    def load(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    memory = torch.load(path, map_location=lambda storage, loc: strage)
                    self.extend(memory)
                    os.remove(path)
                    lock.release()
                    gc.collect()
                    return
                else:
                    sleep(np.random.random())
            return
    
    def update_priority(self, index, priority):
        self.memory['priority'][index] = priority

    def get_stacked_state(self, index):
        stack_count = self.memory['stack_count'][index]
        start_index = index - (self.n_stacks - stack_count)
        if start_index < 0:
            start_index = self.memory_size + start_index
        stack_index = [start_index for _ in range(stack_count)] + [(start_index+1+i)%self.memory_size for i in range(self.n_stacks-stack_count)]
        stacked_state = np.concatenate([self.memory['state'][i] for i in stack_index])
        return stacked_state

    def sample(self, device='cpu'):
        priority = self.memory['priority'][:self.size]
        index = WeightedRandomSampler(
            priority / np.sum(priority),
            self.batch_size,
            replacement=True)
        index = np.array(list(index))
        #index = np.random.randint(0, self.size, self.batch_size)
        next_index = (index + self.n_step) % self.memory_size

        batch = dict()
        batch['state'] = np.stack([self.get_stacked_state(i) for i in index])
        batch['next_state'] = np.stack([self.get_stacked_state(i) for i in next_index])
        batch['action'] = self.memory['action'][index]
        batch['reward'] = self.memory['reward'][index]
        batch['done'] = self.memory['done'][index]

        for key in ['state', 'next_state']:
            batch[key] = batch[key].astype(np.float32) / 255.
        
        for key in batch.keys():
            batch[key] = torch.FloatTensor(batch[key]).to(device)
        batch['action'] = batch['action'].long()

        self.beta = min(self.beta + self.beta_step, 1)
        weights = (self.size * priority[index]) ** (-self.beta)
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(device)

        return batch, index, weights
    
    def indexing_sample(self, start_index, last_index, device='cpu'):
        index = np.array([i for i in range(start_index, last_index)])
        next_index = index + self.n_step

        batch = dict()
        batch['state'] = np.stack([self.get_stacked_state(i) for i in index])
        batch['next_state'] = np.stack([self.get_stacked_state(i%self.memory_size) for i in next_index])
        batch['action'] = self.memory['action'][index]
        batch['reward'] = self.memory['reward'][index]
        batch['done'] = self.memory['done'][index]

        batch['state'] = batch['state'].astype(np.float32) / 255.
        batch['next_state'] = batch['next_state'].astype(np.float32) / 255.
        return batch, index