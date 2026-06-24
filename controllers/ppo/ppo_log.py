import os
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco
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
    "det_abs_final_y_mean", "det_heading_bias_deg",
    "stoch_distance_mean", "stoch_distance_std", "stoch_steps_mean", "stoch_steps_std",
    "stoch_fell_count", "stoch_survival_rate", "stoch_reward_mean",
    "stoch_abs_final_y_mean", "stoch_heading_bias_deg",
    # single no-reset-noise deterministic run (isolates the policy's own heading bias)
    "noiseless_distance", "noiseless_final_y", "noiseless_mean_abs_y",
    "noiseless_heading_bias_deg", "noiseless_mean_abs_yaw_deg",
]


def _yaw_deg(qpos):
    """Heading (deg) of the body +x axis in the world xy-plane. 0 = facing +x."""
    forward = np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([1.0, 0.0, 0.0]), qpos[3:7])
    return np.degrees(np.arctan2(forward[1], forward[0]))


def _collect_paths(model, n_episodes, n_steps, deterministic):
    """Run n_episodes, return list of (xs, ys) per episode."""
    env = DogEnv()
    paths = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        xs = [float(env.unwrapped.data.qpos[0])]
        ys = [float(env.unwrapped.data.qpos[1])]
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            xs.append(float(env.unwrapped.data.qpos[0]))
            ys.append(float(env.unwrapped.data.qpos[1]))
            if terminated or truncated:
                break
        paths.append((xs, ys))
    env.close()
    return paths


def _plot_trajectory(stoch_paths, det_paths, version, version_dir, n_episodes,
                     noiseless_paths=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")

    path_groups = [stoch_paths, det_paths]
    if noiseless_paths:
        path_groups.append(noiseless_paths)
    all_xs = [x for paths in path_groups for xs, ys in paths for x in xs]
    all_ys = [y for paths in path_groups for xs, ys in paths for y in ys]

    for i, (xs, ys) in enumerate(stoch_paths):
        ax.plot(xs, ys, color="steelblue", linewidth=1.0, alpha=0.3,
                label=f"stochastic (n={n_episodes})" if i == 0 else None)

    for i, (xs, ys) in enumerate(det_paths):
        ax.plot(xs, ys, color="crimson", linewidth=2.0, alpha=0.7,
                label=f"deterministic (n={n_episodes})" if i == 0 else None)

    if noiseless_paths:
        for i, (xs, ys) in enumerate(noiseless_paths):
            ax.plot(xs, ys, color="black", linewidth=2.5, alpha=0.9,
                    label="deterministic, no reset noise (n=1)" if i == 0 else None)

    ax.plot(stoch_paths[0][0][0], stoch_paths[0][1][0], "go", markersize=8, label="start (all)")
    ax.plot([min(all_xs), max(all_xs)], [0, 0], "--", color="gray",
            linewidth=1, alpha=0.5, label="ideal (y=0)")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(f"v{version} — deterministic (crimson) vs stochastic (blue) overlay")
    ax.legend(loc="upper left", framealpha=0.9)
    out = os.path.join(version_dir, f"trajectory_v{version}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _rollout(model, env, n_steps, deterministic):
    """One episode. Returns a dict with trajectory + drift/heading metrics.

    final_y         signed y at the end (m) — which side it drifted to.
    mean_abs_y      path-averaged |y| (m) — overall off-centerness.
    heading_bias    mean signed yaw (deg) — open-loop heading bias (the curve cause).
    mean_abs_yaw    path-averaged |yaw| (deg) — heading-error magnitude.
    xs, ys          full trajectory, reused for the plot overlay.
    """
    obs, _ = env.reset()
    xs, ys, yaws, rewards = [], [], [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        qpos = env.unwrapped.data.qpos
        xs.append(float(qpos[0]))
        ys.append(float(qpos[1]))
        yaws.append(_yaw_deg(qpos))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    yaws = np.array(yaws) if yaws else np.array([0.0])
    ys_arr = np.array(ys) if ys else np.array([0.0])
    return {
        "xs": xs, "ys": ys,
        "distance":     xs[-1] - xs[0] if xs else 0.0,
        "steps":        len(xs),
        "reward":       sum(rewards),
        "final_y":      ys[-1] if ys else 0.0,
        "mean_abs_y":   float(np.mean(np.abs(ys_arr))),
        "heading_bias": float(np.mean(yaws)),
        "mean_abs_yaw": float(np.mean(np.abs(yaws))),
    }


def _run_episodes(model, n_episodes, n_steps, deterministic):
    """Run n_episodes silent rollouts, return a stats dict."""
    env = DogEnv()
    runs = [_rollout(model, env, n_steps, deterministic) for _ in range(n_episodes)]
    env.close()

    step_counts = np.array([r["steps"] for r in runs])
    distances   = np.array([r["distance"] for r in runs])
    fell_count = int(np.sum(step_counts < n_steps))
    return {
        "distance_mean": round(float(distances.mean()), 4),
        "distance_std":  round(float(distances.std()), 4),
        "steps_mean":    round(float(step_counts.mean()), 1),
        "steps_std":     round(float(step_counts.std()), 1),
        "fell_count":    fell_count,
        "survival_rate": round(1 - fell_count / n_episodes, 2),
        "reward_mean":   round(float(np.mean([r["reward"] for r in runs])), 2),
        "abs_final_y_mean": round(float(np.mean([abs(r["final_y"]) for r in runs])), 3),
        "heading_bias_deg": round(float(np.mean([r["heading_bias"] for r in runs])), 2),
    }


def _write_summary(row):
    """Upsert one row into summary.csv keyed by version (re-running replaces, not appends).
    If an existing file has incompatible columns, archive it once instead of clobbering."""
    rows = []
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == CSV_FIELDS:
                rows = [r for r in reader if str(r.get("version")) != str(row["version"])]
            else:
                archive = os.path.join(EVAL_DIR, "summary_legacy.csv")
                if not os.path.exists(archive):
                    os.rename(SUMMARY_CSV, archive)
                    print(f"  summary.csv columns changed; archived old file to {archive}")
                else:
                    os.remove(SUMMARY_CSV)
    rows.append(row)
    rows.sort(key=lambda r: int(r["version"]) if str(r["version"]).isdigit() else 1 << 30)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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

    stoch_paths = _collect_paths(model, n_episodes, n_steps, deterministic=False)
    det_paths   = _collect_paths(model, n_episodes, n_steps, deterministic=True)

    # single deterministic, no-reset-noise run: reused for the plot overlay AND its metrics
    nl_env = DogEnv(reset_noise_scale=0.0)
    noiseless = _rollout(model, nl_env, n_steps, deterministic=True)
    nl_env.close()
    noiseless_paths = [(noiseless["xs"], noiseless["ys"])]

    traj_path = _plot_trajectory(stoch_paths, det_paths, version, version_dir, n_episodes,
                                 noiseless_paths=noiseless_paths)

    det_stats   = _run_episodes(model, n_episodes, n_steps, deterministic=True)
    stoch_stats = _run_episodes(model, n_episodes, n_steps, deterministic=False)
    noiseless_stats = {
        "distance":         round(noiseless["distance"], 4),
        "final_y":          round(noiseless["final_y"], 3),
        "mean_abs_y":       round(noiseless["mean_abs_y"], 3),
        "heading_bias_deg": round(noiseless["heading_bias"], 2),
        "mean_abs_yaw_deg": round(noiseless["mean_abs_yaw"], 2),
    }

    metrics = {
        "version":      version,
        "n_episodes":   n_episodes,
        "deterministic": det_stats,
        "stochastic":    stoch_stats,
        "noiseless":     noiseless_stats,
    }

    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    row = {"version": version, "n_episodes": n_episodes}
    for k, v in det_stats.items():
        row[f"det_{k}"] = v
    for k, v in stoch_stats.items():
        row[f"stoch_{k}"] = v
    for k, v in noiseless_stats.items():
        row[f"noiseless_{k}"] = v
    _write_summary(row)

    d = det_stats
    s = stoch_stats
    n = noiseless_stats
    print(f"v{version} ({n_episodes} eps) — deterministic: "
          f"dist {d['distance_mean']:.3f} ± {d['distance_std']:.3f} m, "
          f"survived {d['survival_rate']*100:.0f}%, "
          f"|final_y| {d['abs_final_y_mean']:.2f} m, bias {d['heading_bias_deg']:+.1f}°")
    print(f"           stochastic:   "
          f"dist {s['distance_mean']:.3f} ± {s['distance_std']:.3f} m, "
          f"survived {s['survival_rate']*100:.0f}%, "
          f"|final_y| {s['abs_final_y_mean']:.2f} m, bias {s['heading_bias_deg']:+.1f}°")
    print(f"           no-noise det: dist {n['distance']:.2f} m, "
          f"final_y {n['final_y']:+.2f} m, heading bias {n['heading_bias_deg']:+.1f}°, "
          f"mean|yaw| {n['mean_abs_yaw_deg']:.1f}°")
    print(f"  trajectory -> {traj_path}")
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
