import gymnasium as gym
import numpy as np
import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import PPOAgent
from utils.buffers import RolloutBuffer 
from utils.preprocessing import stack_frames, frame_stack_size


ENV_NAME = 'CarRacing-v3'
CONTINUOUS_ACTIONS = True
LOAD_CHECKPOINT = True
LOAD_PREFIX = 'CarRacing/PPO/resultNotLRdecay/ppo_carracing_best_avg'

N = 4096
BATCH_SIZE = 128 
N_EPOCHS = 4     
INITIAL_ALPHA = 1e-4 
GAMMA = 0.99       
GAE_LAMBDA = 0.95  
POLICY_CLIP = 0.2  
VF_COEF = 0.5      
ENT_COEF = 0.001   
USE_LR_DECAY = True
FINAL_LR_FACTOR = 0.05
DECAY_TOTAL_LEARN_ITERS = 300
NUM_EVAL_EPISODES = 5

env = gym.make(ENV_NAME, continuous=CONTINUOUS_ACTIONS, render_mode='human')

IMG_HEIGHT = 84
IMG_WIDTH = 84
INPUT_DIMS = (frame_stack_size, IMG_HEIGHT, IMG_WIDTH)
N_ACTIONS = env.action_space.shape[0]
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high
print(f"Environment: {ENV_NAME}")
print(f"Input Dimensions: {INPUT_DIMS}")
print(f"Action Space Dim: {N_ACTIONS}")

# Agent initialization
agent = PPOAgent(
    n_actions=N_ACTIONS,
    input_dims=INPUT_DIMS,
    action_low=ACTION_LOW,
    action_high=ACTION_HIGH,
    initial_alpha=INITIAL_ALPHA,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    policy_clip=POLICY_CLIP,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    vf_coef=VF_COEF,
    ent_coef=ENT_COEF,
    use_lr_decay=USE_LR_DECAY,
    final_lr_factor=FINAL_LR_FACTOR,
    decay_total_learn_iters=DECAY_TOTAL_LEARN_ITERS
)

if LOAD_CHECKPOINT and LOAD_PREFIX is not None:
    print(f"--- Loading model checkpoint: {LOAD_PREFIX} ---")
    agent.load_models(LOAD_PREFIX) 
    agent.set_to_eval_mode()
    print(f"--- Model loaded successfully ---")
else:
    print("ERROR: No checkpoint specified to load for recording.")
    exit() 

all_rewards = []
for i in range(NUM_EVAL_EPISODES):
    start_time_ep = time.time()
    observation, info = env.reset() 
    stacked_observation = stack_frames(observation, is_new_episode=True)
    done = False
    truncated = False
    score = 0
    episode_steps = 0

    while not done and not truncated:
        agent.set_to_eval_mode()
        _, action_env, _, _ = agent.choose_action(stacked_observation)
        observation, reward, done, truncated, info = env.step(action_env)
        score += reward
        episode_steps += 1
        stacked_observation = stack_frames(observation, is_new_episode=False)
    end_time_ep = time.time()
    episode_time = end_time_ep - start_time_ep
    all_rewards.append(score)
    print(f"Episode {i+1}/{NUM_EVAL_EPISODES} finished in {episode_steps} steps. Score: {score:.2f}. Time: {episode_time:.1f}s")
