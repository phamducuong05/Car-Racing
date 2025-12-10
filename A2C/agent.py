import numpy as np
import torch
import torch.nn.functional as F


class A2C:
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        critic_weight=0.5,
        entropy_weight=0.01,
    ):
        self.model = model
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def learn(
        self,
        batch_states,
        batch_actions,
        batch_returns,
        batch_advantages,
    ):
        new_action_log_probs, entropies, new_state_values = self.model.evaluate(batch_states, batch_actions)

        entropy = entropies.mean()

        # Calculate Actor loss
        actor_loss = - (new_action_log_probs * batch_advantages).mean()

        # Calculate Critic loss
        critic_loss = F.mse_loss(new_state_values, batch_returns)

        # Calculate total loss
        loss = (
            actor_loss
            + self.critic_weight * critic_loss
            - self.entropy_weight * entropy
        )

        # Update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
