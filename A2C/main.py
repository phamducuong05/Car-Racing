import os

import gymnasium as gym
import numpy as np
import torch

from envwrapper import EnvWrapper
from model import ActorCritic
from util import save_gif


def main():
    model_dir = "CarRacing/A2C/results/model_a2c"
    env_id = "CarRacing-v3"
    eval_seed = 315
    num_eval_tracks = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    if not os.path.exists(model_dir):
        print(f"ERROR: Checkpoint directory not found at {model_dir}")
        exit(1)

    print(
        f"Generating {num_eval_tracks} evaluation track seeds using master seed {eval_seed}..."
    )
    rng = np.random.default_rng(seed=eval_seed)

    track_seeds = rng.integers(low=0, high=2**31 - 1, size=num_eval_tracks)

    env = EnvWrapper(gym.make(env_id, domain_randomize=False, render_mode="human"))

    model = ActorCritic(env.observation_space.shape, env.action_space.shape[0]).to(
        device
    )
    model.eval()

    highest_score_ever = -float("inf")
    best_avg_score = -float("inf")
    best_fname = ""

    print(f"Evaluating checkpoints in {model_dir}...")

    fname = "best_a2c.pt"
    ckpt_path = os.path.join(model_dir, fname)
    print(f"  Evaluating {fname} ... ", flush=True)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    current_model_total_score = 0
    num_valid_runs = 0

    for i, seed in enumerate(track_seeds):
        frames = []
        score = 0
        terminated = False
        truncated = False

        try:
            state, _ = env.reset(seed=seed.item())
            step = 0
            max_eval_steps = 1000
            while not terminated and not truncated and step < max_eval_steps:
                current_frame = env.render()
                if current_frame is not None:
                    frames.append(current_frame)

                state_tensor = torch.tensor(
                    np.expand_dims(state, axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    action, _, _ = model(state_tensor, determinstic=True)
                    action_np = action.cpu().numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(action_np)

                score += reward
                state = next_state
                step += 1

            if step > 0:
                current_model_total_score += score
                num_valid_runs += 1

                if score > highest_score_ever:
                    print(
                        f"\n    New highest single score: {score:.2f} from {fname} on seed {seed}"
                    )
                    highest_score_ever = score

        except Exception as e:
            print(
                f"\n    Error during evaluation run with seed {seed} for {fname}: {e}"
            )

    env.close()  # Đóng môi trường sau khi đánh giá xong

    if best_fname:
        print("-" * 30)
        print(f"Evaluation finished.")
        print(f"The best model checkpoint is: {best_fname}")
        print(f"With an average score of:   {best_avg_score:.3f}")
        print(
            f"Highest single track score found: {highest_score_ever:.2f} (best_play.gif)"
        )
    else:
        print("-" * 30)
        print(
            "Evaluation finished, but no valid checkpoints were successfully evaluated."
        )


if __name__ == "__main__":
    main()
