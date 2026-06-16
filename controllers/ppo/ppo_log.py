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


def _rollout(model, env, n_steps):
    """One episode. Returns (distance, steps, total_reward)."""
    obs, _ = env.reset()
    positions, rewards = [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(float(env.unwrapped.data.qpos[0]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    distance = positions[-1] - positions[0] if positions else 0.0
    return distance, len(positions), sum(rewards)


def evaluate(version, n_episodes=10, n_steps=1000, record=True):
    """Evaluate dog_ppo_v{version} over n_episodes. Records ONE video,
    averages metrics over all episodes, writes metrics.json + summary.csv row."""
    model_path = os.path.join(MODEL_DIR, f"dog_ppo_v{version}")
    version_dir = os.path.join(EVAL_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    model = PPO.load(model_path)

    # --- one recorded episode for the video ---
    if record:
        rec_env = DogEnv(render_mode="rgb_array")
        rec_env = RecordVideo(
            rec_env,
            video_folder=version_dir,
            name_prefix=f"dog_ppo_v{version}",
            episode_trigger=lambda ep: True,
        )
        _rollout(model, rec_env, n_steps)
        rec_env.close()   # finalizes the mp4

    # --- n_episodes silent rollouts for statistics ---
    eval_env = DogEnv()  # no rendering, fast
    distances, step_counts, total_rewards = [], [], []
    for _ in range(n_episodes):
        d, s, r = _rollout(model, eval_env, n_steps)
        distances.append(d)
        step_counts.append(s)
        total_rewards.append(r)
    eval_env.close()

    distances = np.array(distances)
    step_counts = np.array(step_counts)
    fell_count = int(np.sum(step_counts < n_steps))

    metrics = {
        "version": version,
        "n_episodes": n_episodes,
        "distance_mean": round(float(distances.mean()), 4),
        "distance_std": round(float(distances.std()), 4),
        "steps_mean": round(float(step_counts.mean()), 1),
        "steps_std": round(float(step_counts.std()), 1),
        "fell_count": fell_count,            # how many of N episodes fell early
        "survival_rate": round(1 - fell_count / n_episodes, 2),
        "reward_mean": round(float(np.mean(total_rewards)), 2),
    }

    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"v{version} over {n_episodes} eps: "
          f"dist {distances.mean():.3f} ± {distances.std():.3f} m, "
          f"steps {step_counts.mean():.0f} ± {step_counts.std():.0f}, "
          f"survived {metrics['survival_rate']*100:.0f}% of episodes")
    print(f"  -> {version_dir}")
    return metrics


if __name__ == "__main__":
    import argparse
    os.makedirs(EVAL_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    evaluate(args.version, n_episodes=args.episodes, record=not args.no_video)

    # copy and paste to run, add version at end (optional: --episodes N, --no-video):
    # python3 controllers/ppo/ppo_log.py