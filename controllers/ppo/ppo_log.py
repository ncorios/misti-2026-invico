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

CSV_FIELDS = [
    "version", "n_episodes",
    "det_distance_mean", "det_distance_std", "det_steps_mean", "det_steps_std",
    "det_fell_count", "det_survival_rate", "det_reward_mean",
    "stoch_distance_mean", "stoch_distance_std", "stoch_steps_mean", "stoch_steps_std",
    "stoch_fell_count", "stoch_survival_rate", "stoch_reward_mean",
]


def _rollout(model, env, n_steps, deterministic):
    """One episode. Returns (distance, steps, total_reward)."""
    obs, _ = env.reset()
    positions, rewards = [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(float(env.unwrapped.data.qpos[0]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    distance = positions[-1] - positions[0] if positions else 0.0
    return distance, len(positions), sum(rewards)


def _run_episodes(model, n_episodes, n_steps, deterministic):
    """Run n_episodes silent rollouts, return a stats dict."""
    env = DogEnv()
    distances, step_counts, total_rewards = [], [], []
    for _ in range(n_episodes):
        d, s, r = _rollout(model, env, n_steps, deterministic)
        distances.append(d)
        step_counts.append(s)
        total_rewards.append(r)
    env.close()

    distances = np.array(distances)
    step_counts = np.array(step_counts)
    fell_count = int(np.sum(step_counts < n_steps))
    return {
        "distance_mean": round(float(distances.mean()), 4),
        "distance_std":  round(float(distances.std()), 4),
        "steps_mean":    round(float(step_counts.mean()), 1),
        "steps_std":     round(float(step_counts.std()), 1),
        "fell_count":    fell_count,
        "survival_rate": round(1 - fell_count / n_episodes, 2),
        "reward_mean":   round(float(np.mean(total_rewards)), 2),
    }


def evaluate(version, n_episodes=10, n_steps=1000, record=True):
    """Dual-mode eval: deterministic + stochastic. Records ONE deterministic video."""
    model_path = os.path.join(MODEL_DIR, f"dog_ppo_v{version}")
    version_dir = os.path.join(EVAL_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    model = PPO.load(model_path)

    # one deterministic recorded episode
    if record:
        rec_env = DogEnv(render_mode="rgb_array")
        rec_env = RecordVideo(
            rec_env,
            video_folder=version_dir,
            name_prefix=f"dog_ppo_v{version}",
            episode_trigger=lambda ep: True,
        )
        _rollout(model, rec_env, n_steps, deterministic=True)
        rec_env.close()

    det_stats   = _run_episodes(model, n_episodes, n_steps, deterministic=True)
    stoch_stats = _run_episodes(model, n_episodes, n_steps, deterministic=False)

    metrics = {
        "version":      version,
        "n_episodes":   n_episodes,
        "deterministic": det_stats,
        "stochastic":    stoch_stats,
    }

    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- summary.csv: rename old file if columns don't match, then append ---
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="") as f:
            existing_fields = next(csv.reader(f), [])
        if existing_fields != CSV_FIELDS:
            old_path = os.path.join(EVAL_DIR, "summary_single_episode.csv")
            os.rename(SUMMARY_CSV, old_path)
            print(f"  Old summary.csv columns don't match new format; renamed to {old_path}")

    write_header = not os.path.exists(SUMMARY_CSV)
    row = {"version": version, "n_episodes": n_episodes}
    for k, v in det_stats.items():
        row[f"det_{k}"] = v
    for k, v in stoch_stats.items():
        row[f"stoch_{k}"] = v

    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    d = det_stats
    s = stoch_stats
    print(f"v{version} ({n_episodes} eps) — deterministic: "
          f"dist {d['distance_mean']:.3f} ± {d['distance_std']:.3f} m, "
          f"survived {d['survival_rate']*100:.0f}%")
    print(f"           stochastic:   "
          f"dist {s['distance_mean']:.3f} ± {s['distance_std']:.3f} m, "
          f"survived {s['survival_rate']*100:.0f}%")
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
    # python3 controllers/ppo/ppo_log.py <version>
