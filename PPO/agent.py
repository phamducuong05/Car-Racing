import torch
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from torch.optim.lr_scheduler import LinearLR
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import ActorNetwork, CriticNetwork
from utils.buffers import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        n_actions,
        input_dims,
        action_low,
        action_high,
        initial_alpha=0.0001, 
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        vf_coef=0.5,
        ent_coef=0.01,
        use_lr_decay=True,
        final_lr_factor=0.05,
        decay_total_learn_iters= 1000
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.use_lr_decay = use_lr_decay


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.action_low = self.action_low.to(self.device)
        self.action_high = self.action_high.to(self.device)

        self.actor = ActorNetwork(input_dims, n_actions, initial_alpha)
        self.critic = CriticNetwork(input_dims, initial_alpha)
        self.memory = RolloutBuffer(batch_size)

        self.actor_scheduler = None
        self.critic_scheduler = None
        if self.use_lr_decay:
            print(f"Start LR={initial_alpha:.1E}, End Factor={final_lr_factor}, Total Learn Iters={decay_total_learn_iters}")
            self.actor_scheduler = LinearLR(
                self.actor.optimizer,
                start_factor=1.0,
                end_factor=final_lr_factor,
                total_iters=decay_total_learn_iters
            )
            self.critic_scheduler = LinearLR(
                self.critic.optimizer,
                start_factor=1.0,
                end_factor=final_lr_factor,
                total_iters=decay_total_learn_iters
            )
        else:
             print(f"Not using LR Decay. Fixed LR={initial_alpha:.1E}")

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, state):
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            dist = self.actor(state_tensor)
            action_sampled = dist.sample()
            log_prob = dist.log_prob(action_sampled).sum(axis=-1)
            value = self.critic(state_tensor)
            action_clipped = torch.clamp(action_sampled, self.action_low, self.action_high)

        self.actor.train()
        self.critic.train()

        action_to_env = action_clipped.cpu().numpy()[0]
        action_stored = action_sampled.cpu().numpy()[0]
        log_prob_item = log_prob.item()
        value_item = value.squeeze().item()

        return value_item, action_to_env, log_prob_item, action_stored

    def learn(self):
        self.actor.train()
        self.critic.train()

        states_np, actions_np, old_probs_np, vals_np, rewards_np, dones_np, batches = \
            self.memory.generate_batches()

        advantages = np.zeros(len(rewards_np), dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards_np))):
            if t == len(rewards_np) - 1:
                next_non_terminal = 1.0 - dones_np[t]
                next_value = vals_np[t]
            else:
                next_non_terminal = 1.0 - dones_np[t]
                next_value = vals_np[t+1]

            delta = rewards_np[t] + self.gamma * next_value * next_non_terminal - vals_np[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + vals_np

        states_tensor = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions_np, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.tensor(old_probs_np, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        for epoch in range(self.n_epochs):
            for batch_indices in batches:
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                batch_adv_mean = batch_advantages.mean()
                batch_adv_std = batch_advantages.std() + 1e-8
                batch_advantages_norm = (batch_advantages - batch_adv_mean) / batch_adv_std

                batch_returns = returns_tensor[batch_indices]

                dist = self.actor(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().mean()

                prob_ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = prob_ratio * batch_advantages_norm
                surr2 = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages_norm
                actor_loss = -torch.min(surr1, surr2).mean()

                new_values = self.critic(batch_states).squeeze(-1)
                critic_loss = F.mse_loss(new_values, batch_returns)

                total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_models(self, filename_prefix='ppo_carracing'):
        torch.save(self.actor.state_dict(), f'{filename_prefix}_actor.pth')
        torch.save(self.critic.state_dict(), f'{filename_prefix}_critic.pth')
        print(f"Models saved with prefix: {filename_prefix}")

    def load_models(self, filename_prefix='ppo_carracing'):
        try:
            self.actor.load_state_dict(torch.load(f'{filename_prefix}_actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{filename_prefix}_critic.pth', map_location=self.device))
            print(f"Models loaded from prefix: {filename_prefix}")
            self.actor.eval()
            self.critic.eval()
        except FileNotFoundError:
            print(f"Error: Model files not found with prefix: {filename_prefix}")
        except Exception as e:
            print(f"Error loading models: {e}")

    def set_to_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.actor.train()
        self.critic.train()
