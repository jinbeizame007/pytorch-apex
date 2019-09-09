import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import fasteners
import os
from time import sleep


class QNet(nn.Module):
    def __init__(self, path, device='cpu'):
        super(QNet, self).__init__()
        self.path = path
        self.device = device

        self.vis_layers = nn.Sequential(
            # (84, 84, *) -> (20, 20, 16)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            # (20, 20, 16) -> (9, 9, 32)
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            # (9, 9, 32) -> (7, 7, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
            #Flatten(),
        )

        self.l1 = nn.Sequential(
            # (7 * 7 * 64, ) -> (512, )
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True)
        )

        self.l2 = nn.Sequential(
            nn.Linear(4, 50),
            nn.ReLU(True)
        )

        self.l3 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(True)
        )

        self.l4 = nn.Linear(50, 2)

        self.val = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

    def forward(self, state):
        """
        state = torch.FloatTensor(state).view(-1,4).to(self.device)
        h = self.l2(state)
        h = self.l3(h)
        val = self.val(h)
        adv = self.adv(h)
        q_val = val + adv - adv.mean(1, keepdim=True)
        return q_val
        """

        h = self.vis_layers(state).view(-1,7*7*64)
        h = self.l1(h)

        # V
        V = self.val(h)
        # A
        A = self.adv(h)
        # Q
        Q = V + A - A.mean(1, keepdim=True)

        return Q
    
    def save(self):
        lock = fasteners.ReaderWriterLock()
        while True:
            try:
                with lock.write_lock():
                    torch.save(self.state_dict(), self.path)
                return
            except:
                sleep(np.random.random()+1)
    
    def load(self):
        lock = fasteners.ReaderWriterLock()
        while True:
            try:
                with lock.read_lock():
                    state_dict = torch.load(self.path)
                self.load_state_dict(state_dict)
                return
            except:
                sleep(np.random.random()+1)
    
    """
    def save(self):
        lock = fasteners.InterProcessLock(self.path)
        while True:
            #try:
            if lock.acquire(blocking=False):
                torch.save(self.state_dict(), self.path)
                lock.release()
                return
            #except:
            sleep(0.2)
    
    def load(self):
        lock = fasteners.InterProcessLock(self.path)
        while True:
            #try:
            if lock.acquire(blocking=False) and os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
                state_dict = torch.load(self.path)
                self.load_state_dict(state_dict)
                lock.release()
                return
            #except:
            sleep(0.2)
    """