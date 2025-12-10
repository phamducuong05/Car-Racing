import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import math
import cv2
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.buffers import ReplayMemory
from model import DQN

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DQNAgent:
    def __init__(
        self,
        n_actions,
        input_shapes,
        BATCH_SIZE = 64,         
        GAMMA = 0.99,            
        EPS_START = 0.9,          
        EPS_END = 0.05,         
        EPS_DECAY = 10000,        
        LEARNING_RATE = 1e-4,
        ):
        self.n_actions = n_actions
        self.input_shapes = input_shapes
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.alpha = LEARNING_RATE
        self.steps_done = 0
    
        self.replay_buffer = ReplayMemory(10000)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_shapes, n_actions).to(device=self.device)
        self.target_net = DQN(input_shapes, n_actions).to(device=self.device)
        self.optimizer =  optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    

    def select_action(self, state):
        """Selects an action using Epsilon-Greedy strategy."""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_index = q_values.max(1)[1].item()
                return action_index
        else:
            return random.randrange(self.n_actions)
        
    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return 
        
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0) 
                                           for s, d in zip(batch.next_state, batch.done) if not d])
        
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0) for s in batch.state])
        action_batch = torch.tensor([[a] for a in batch.action], dtype=torch.long, device=self.device)        
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()       
        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()    
        