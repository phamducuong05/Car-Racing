import gymnasium as gym
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import stack_frames, frame_stack_size
from agent import PPOAgent


"""Configuration"""
ENV_NAME = 'CarRacing-v3'
CONTINUOUS_ACTIONS = True 

# Training Hyperparameters
N = 8192
BATCH_SIZE = 256
N_EPOCHS = 4
INITIAL_ALPHA = 0.0001    
GAMMA = 0.99
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
VF_COEF = 0.5 
ENT_COEF = 0.005

USE_LR_DECAY = False        
FINAL_LR_FACTOR = 0.05      # LR cuối = initial_alpha * 0.05 = 5e-6
DECAY_TOTAL_LEARN_ITERS = 1000 

TOTAL_GAMES = 3000 
SAVE_FREQ = 50
LOAD_CHECKPOINT = False
CHECKPOINT_PREFIX = 'ppo_carracing'


env = gym.make(ENV_NAME, continuous=CONTINUOUS_ACTIONS, render_mode=None)


IMG_HEIGHT = 84
IMG_WIDTH = 84
INPUT_DIMS = (frame_stack_size, IMG_HEIGHT, IMG_WIDTH)
N_ACTIONS = env.action_space.shape[0]
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high


print(f"Environment: {ENV_NAME}")
print(f"Input Dimensions: {INPUT_DIMS}")
print(f"Action Space Dim: {N_ACTIONS}")
print(f"Action Low: {ACTION_LOW}")
print(f"Action High: {ACTION_HIGH}")


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

if LOAD_CHECKPOINT:
    print(f"Loading models from prefix: {CHECKPOINT_PREFIX}")
    agent.load_models(CHECKPOINT_PREFIX)

# Tracking
score_history = []
avg_score_history = []
learn_iters = 0
n_steps = 0
best_score = -np.inf 

print(f"\nStarting training for {TOTAL_GAMES} games...")
print(f"Rollout Length (N): {N}, Batch Size: {BATCH_SIZE}, Epochs: {N_EPOCHS}\n")
if USE_LR_DECAY:
    current_lr_display = agent.actor.optimizer.param_groups[0]['lr']
    print(f"Using LR Decay: Start={INITIAL_ALPHA:.1E}, Current={current_lr_display:.1E}, End Factor={FINAL_LR_FACTOR}, Total Learn Iters={DECAY_TOTAL_LEARN_ITERS}")

# Training loop
try:
    for i in range(TOTAL_GAMES):
        raw_observation, info = env.reset()
        stacked_observation = stack_frames(raw_observation, is_new_episode=True)
        done = False
        truncated = False
        score = 0
        episode_steps = 0

        while not done and not truncated:
            value, action_env, log_prob, action_mem = agent.choose_action(stacked_observation)
            next_raw_observation, reward, done, truncated, info = env.step(action_env)
            n_steps += 1
            episode_steps += 1
            score += reward
            stacked_next_observation = stack_frames(next_raw_observation, is_new_episode=False)
            agent.remember(stacked_observation, action_mem, log_prob, value, reward, done or truncated)
            if agent.memory.__len__() >= N:
                current_lr_display = agent.actor.optimizer.param_groups[0]['lr']
                log_std_val = "N/A"
                try:
                     log_std_val = f"{agent.actor.fc[-1].log_std.data.mean().item():.3f}"
                except AttributeError: pass
                print(f"\nLearn Iter: {learn_iters + 1}, LR: {current_lr_display:.2E}, Avg log_std: {log_std_val})")
                agent.set_to_train_mode() 
                agent.learn() 
                learn_iters += 1
                if agent.actor_scheduler:
                    agent.actor_scheduler.step()
                if agent.critic_scheduler:
                    agent.critic_scheduler.step()
                print(f"Learning update complete (iteration {learn_iters}).")
            stacked_observation = stacked_next_observation
            

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)


        if avg_score > best_score:
             best_score = avg_score
             print(f"New best average score: {best_score:.2f}")
             agent.save_models(f"{CHECKPOINT_PREFIX}_best_avg")

        print(f'Game: {i+1}/{TOTAL_GAMES} | Ep Steps: {episode_steps:4d} | Score: {score:8.2f} | Avg Score(100): {avg_score:8.2f}')

        # Periodic Checkpoint Saving
        if (i + 1) % SAVE_FREQ == 0 and (i+1) > 0:
            print(f"\n--- Saving checkpoint at game {i+1} ---")
            agent.save_models(f"{CHECKPOINT_PREFIX}_game_{i+1}")
            print("--- Checkpoint saved ---\n")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:

    print("\nTraining finished.")
    env.close()
    agent.save_models(f"{CHECKPOINT_PREFIX}_final")
    print("Final models saved.")
