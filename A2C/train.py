import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from envwrapper import EnvWrapper
from model import ActorCritic
from agent import A2C
from util import fix_random_seeds, play


def calculate_discounted_returns(b_rewards, discount_factor):
    discounted_returns = np.zeros_like(b_rewards, dtype=np.float32)  
    running_add = 0
    for i in reversed(range(b_rewards.shape[0])):
        running_add = b_rewards[i] + discount_factor * running_add
        discounted_returns[i] = running_add
    return discounted_returns


def train(
    env,
    model,
    agent,
    device,
    n_episodes=3000,
    discount_factor=0.99,
    max_steps_per_episode=1000,
):
    if not os.path.exists("A2C/results/model_a2c"):
        os.makedirs("A2C/results/model_a2c")
    if not os.path.exists("A2C/results/plot_a2c"):
        os.makedirs("A2C/results/plot_a2c")

    writer = SummaryWriter(log_dir="A2C/results/plot_a2c")

    max_steps = max_steps_per_episode

    print(f"Training on {device} using A2C...")
    print(f"Max steps per episode: {max_steps}")

    score_history = []
    avg_score_history = []
    best_avg_score = -np.inf
    try:
        for episode in range(n_episodes):
            # Containers to store the trajectory
            states_list = []
            actions_list = []
            action_probs_list = []
            state_values_list = []
            rewards_list = []
            terminated = False
            truncated = False

            state, _ = env.reset()
            current_step = 0

            while not terminated and not truncated and current_step < max_steps:
                state_tensor = torch.tensor(
                    np.expand_dims(state, axis=0), dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    action, action_prob, state_value = model.forward(state_tensor)

                action_np = action.cpu().numpy()[0]
                action_prob_np = action_prob.cpu().numpy()
                state_value_np = state_value.cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action_np)

                # Store the trajectory into the containers
                states_list.append(state)
                actions_list.append(action_np)
                action_probs_list.append(action_prob_np)
                state_values_list.append(state_value_np)
                rewards_list.append(reward)

                state = next_state
                current_step += 1

            step_cnt = len(rewards_list)

            if step_cnt == 0:
                print(
                    f"[Episode {episode + 1:4d}/{n_episodes}] Skipped - Episode ended immediately."
                )
                continue

            b_states = np.array(states_list, dtype=np.float32)
            b_actions = np.array(actions_list, dtype=np.float32)
            b_state_values_np = np.array(state_values_list, dtype=np.float32)
            b_rewards_np = np.array(rewards_list, dtype=np.float32)

            b_states = torch.from_numpy(b_states).to(device)
            b_actions = torch.from_numpy(b_actions).to(device)
            b_state_values = torch.from_numpy(b_state_values_np).to(device)

            b_returns_np = calculate_discounted_returns(b_rewards_np, discount_factor)
            b_returns = torch.from_numpy(b_returns_np.copy()).to(
                device, dtype=torch.float32
            )

            b_advantages = b_returns - b_state_values
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

            b_advantages = b_advantages.detach()

            b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            b_returns = b_returns.detach()

            loss, actor_loss, critic_loss, entropy = agent.learn(
                b_states,
                b_actions,
                b_returns,
                b_advantages,
            )

            # Logging
            total_reward = b_rewards_np.sum()

            score_history.append(total_reward)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            # Save model that has the highest average score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                ckpt_path = f"A2C/results/model_a2c/best_a2c.pt"
                print(f"Saving checkpoint to {ckpt_path}... ", flush=True)
                torch.save(
                    {
                        "episode": episode + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "loss": loss,
                    },
                    ckpt_path,
                )

            print(
                f"[Episode {episode + 1:4d}/{n_episodes}] Steps = {step_cnt}, Loss = {loss:.2f}, ",
                f"Actor Loss = {actor_loss:.2f}, Critic Loss = {critic_loss:.2f}, ",
                f"Entropy = {entropy:.2f}, ",
                f"Total Reward = {total_reward:.2f}, ",
                f"Avg score(100) = {avg_score:.2f}",
            )

            # Store log to TensorBoard
            writer.add_scalar("Loss/Episode", loss, episode + 1)
            writer.add_scalar("Loss/Actor Loss", actor_loss, episode + 1)
            writer.add_scalar("Loss/Critic Loss", critic_loss, episode + 1)
            writer.add_scalar("Performance/Entropy", entropy, episode + 1)
            writer.add_scalar("Performance/Total Reward", total_reward, episode + 1)
            writer.add_scalar("Performance/Episode Length", step_cnt, episode + 1)
            writer.flush()

            # Store checkpoint periodically
            if (episode + 1) % 100 == 0:
                ckpt_path = (
                    f"A2C/results/model_a2c/a2c_checkpoint_{episode + 1:04d}.pt"
                )
                print(f"Saving checkpoint to {ckpt_path}... ", flush=True)
                torch.save(
                    {
                        "episode": episode + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),  # Lưu cả optimizer state
                        "loss": loss,
                    },
                    ckpt_path,
                )
                # Store the video of the checkpoint
                try:
                    play(model, f"Train_episode_{episode + 1:04d}.gif")
                except NameError:
                    pass
                print("Done!")
    except KeyboardInterrupt:
        print("Training interupted by user.")
    finally:
        writer.close()
        # Plot result
        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(len(score_history)),
            score_history,
            label="Episode Score",
            alpha=0.7,
        )
        plt.plot(
            np.arange(len(avg_score_history)),
            avg_score_history,
            label="Avg Score (100 episodes)",
            linewidth=2,
            color="red",
        )
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title(f"Training Progress - A2C on CarRacing-v3")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("results", "a2c_training_plot.png"))
        print(
            f"Training plot saved to {os.path.join("results", "a2c_training_plot.png")}"
        )
        print("Training finished.")


def main():
    seed = 315
    fix_random_seeds(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        env_raw = gym.make(
            "CarRacing-v3", domain_randomize=False, render_mode="rgb_array"
        )
        env = EnvWrapper(env_raw, seed=seed)
    except Exception as e:
        print(f"Error creating env: {e}")
        return

    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0]).to(
        device
    )

    lr = 1e-4
    critic_weight = 0.5
    ent_weight = 0.01
    agent = A2C(
        model, learning_rate=lr, critic_weight=critic_weight, entropy_weight=ent_weight
    )
    print(
        f"Initialized A2C agent with lr={lr}, critic_w={critic_weight}, entropy_w={ent_weight}"
    )

    train(
        env,
        model,
        agent,
        device,
        n_episodes=3000,
        discount_factor=0.99,
        max_steps_per_episode=1000,
    )


if __name__ == "__main__":
    main()
