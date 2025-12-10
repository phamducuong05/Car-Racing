import gymnasium as gym
import torch
import torch.nn as nn
import sys
import os
import time
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import *
from agent import DQNAgent



BATCH_SIZE = 64           # Number of transitions sampled from the replay buffer
GAMMA = 0.99              # Discount factor for future rewards
EPS_START = 0.9           # Starting value of epsilon (exploration rate)
EPS_END = 0.05            # Minimum value of epsilon
EPS_DECAY = 10000         # Controls the rate of epsilon decay (larger value -> slower decay)
TARGET_UPDATE_FREQ = 1000 # How often to update the target network (in steps)
LEARNING_RATE = 1e-4

num_episodes = 1000
max_steps_per_episode = 1000 
episode_rewards = []
episode_durations = []
total_steps = 0
score_history = []         
avg_score_history = []

MODEL_PATH = 'CarRacing/DQN/results64/policy_net_trained.pth' 
env = gym.make('CarRacing-v3', continuous=False, domain_randomize=False, render_mode='human')
dummy_state, _ = env.reset()
initial_stacked_state = stack_frames(dummy_state, is_new_episode=True)
input_shape = initial_stacked_state.shape
n_actions = 5

agent = DQNAgent(n_actions, input_shape, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, LEARNING_RATE)

if not os.path.exists(MODEL_PATH):
    print(f"LỖI: Không tìm thấy file model tại '{MODEL_PATH}'")
    env.close() 
else:
    try:
        agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
        agent.policy_net.eval() 
        for i_episode in range(5):
            observation, info = env.reset()
            current_state = stack_frames(observation, is_new_episode=True)
            total_reward = 0
            steps = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(current_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    action_index = agent.policy_net(state_tensor).max(1)[1].item()
                observation, reward, terminated, truncated, info = env.step(action_index)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                current_state = stack_frames(observation, is_new_episode=False)
            print(f"Episode {i_episode} finished in {steps} steps. Score: {total_reward:.2f}.")

        env.close()
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình tải model hoặc chạy đánh giá: {e}")
        env.close()
