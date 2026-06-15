import os
import csv
import json
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

from dogzilla_env import DogEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
EVAL_DIR = os.path.join(HERE, "ppo_eval")
SUMMARY_CSV = os.path.join(EVAL_DIR, "summary.csv")


def evaluate(version, n_steps=1000, record=True):
    """Roll out saved model dog_ppo_v{version} once. Writes video + metrics
    into ppo_eval/v{version}/, appends a row to summary.csv. One rollout feeds
    both the video and the numbers."""
    model_path = os.path.join(MODEL_DIR, f"dog_ppo_v{version}")
    version_dir = os.path.join(EVAL_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    env = DogEnv(render_mode="rgb_array")
    if record:
        env = RecordVideo(
            env,
            video_folder=version_dir,
            name_prefix=f"dog_ppo_v{version}",
            episode_trigger=lambda ep: True,
        )

    model = PPO.load(model_path)

    obs, _ = env.reset()
    positions, rewards = [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # privileged sim state — fine for eval, never in the policy obs.
        # qpos[0] = base x-position. VERIFY this index matches your model.
        positions.append(float(env.unwrapped.data.qpos[0]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    env.close()   # finalizes the mp4 — mandatory

    steps = len(positions)
    distance = positions[-1] - positions[0]
    metrics = {
        "version": version,
        "distance_m": round(distance, 4),
        "steps_survived": steps,
        "fell_early": steps < n_steps,
        "total_reward": round(sum(rewards), 2),
        "mean_reward": round(float(np.mean(rewards)), 4),
    }

    # per-version metrics.json
    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # append to rolling summary.csv (write header if file is new)
    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"v{version}: {distance:.3f} m over {steps} steps "
          f"({'fell early' if steps < n_steps else 'full episode'}), "
          f"reward {sum(rewards):.1f}")
    print(f"  -> {version_dir}")
    return metrics


if __name__ == "__main__":
    import argparse
    os.makedirs(EVAL_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    evaluate(args.version, record=not args.no_video)