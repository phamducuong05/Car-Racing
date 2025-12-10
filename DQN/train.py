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

env = gym.make('CarRacing-v3', continuous=False, domain_randomize=False, render_mode='rgb_array')
dummy_state, _ = env.reset()
initial_stacked_state = stack_frames(dummy_state, is_new_episode=True)
input_shape = initial_stacked_state.shape
n_actions = 5

agent = DQNAgent(n_actions, input_shape, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, LEARNING_RATE)

for i_episode in range(num_episodes):
    observation, info = env.reset()
    current_state = stack_frames(observation, is_new_episode=True)
    current_episode_reward = 0
    start_time = time.time()
    for t in range(max_steps_per_episode):
        action_index = agent.select_action(current_state) 
        observation, reward, terminated, truncated, _ = env.step(action_index)
        done = terminated or truncated
        current_episode_reward += reward
        next_state = stack_frames(observation, is_new_episode=False)
        agent.replay_buffer.push(current_state, action_index, next_state, reward, done)
        current_state = next_state
        
        agent.optimize_model()
        total_steps += 1

        if total_steps % TARGET_UPDATE_FREQ == 0:
            print(f"Updating target network at step {total_steps}")
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if done:
            break 

    episode_rewards.append(current_episode_reward)
    episode_durations.append(t + 1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_score = np.mean(episode_rewards[-100:])
    avg_score_history.append(avg_score) 
    current_epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.steps_done / EPS_DECAY)
    print(f"Episode {i_episode+1}/{num_episodes} | Duration: {t+1} steps | Reward: {current_episode_reward:.2f} | Avg Reward (last 100): {avg_score:.2f} | Epsilon: {current_epsilon if isinstance(current_epsilon, str) else f'{current_epsilon:.3f}'} | Time: {elapsed_time:.2f}s")


print('Training complete')
torch.save(agent.policy_net.state_dict(), 'policy_net_trained.pth')
env.close()
