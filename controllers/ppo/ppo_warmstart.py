import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import TimeLimit
from dogzilla_env import DogEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


class RewardTermsCallback(BaseCallback):
    """Logs all five reward terms to TensorBoard."""

    def __init__(self):
        super().__init__()
        self._ep_reward_forward = []
        self._ep_reward_survive = []
        self._ep_reward_smoothness = []
        self._ep_reward_stability = []
        self._ep_reward_turning = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "reward_forward" in info:
                self._ep_reward_forward.append(info["reward_forward"])
                self._ep_reward_survive.append(info["reward_survive"])
                self._ep_reward_smoothness.append(info["reward_smoothness"])
                self._ep_reward_stability.append(info["reward_stability"])
                self._ep_reward_turning.append(info["reward_turning"])
        return True

    def _on_rollout_end(self) -> None:
        if self._ep_reward_forward:
            self.logger.record("reward/forward",    np.mean(self._ep_reward_forward))
            self.logger.record("reward/survive",    np.mean(self._ep_reward_survive))
            self.logger.record("reward/smoothness", np.mean(self._ep_reward_smoothness))
            self.logger.record("reward/stability",  np.mean(self._ep_reward_stability))
            self.logger.record("reward/turning",    np.mean(self._ep_reward_turning))
            self._ep_reward_forward.clear()
            self._ep_reward_survive.clear()
            self._ep_reward_smoothness.clear()
            self._ep_reward_stability.clear()
            self._ep_reward_turning.clear()


def make_env():
    env = DogEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def warmstart(from_version, to_version, total_timesteps=2_000_000, n_envs=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if n_envs is None:
        n_envs = max(1, os.cpu_count() - 2)

    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    load_path = os.path.join(MODEL_DIR, f"dog_ppo_v{from_version}")
    model = PPO.load(load_path, env=vec_env)

    
    model.set_logger(configure(os.path.join(HERE, "tb_logs", f"v{to_version}"), ["stdout", "tensorboard"]))

    callback = RewardTermsCallback()
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=callback,
    )

    save_path = os.path.join(MODEL_DIR, f"dog_ppo_v{to_version}")
    model.save(save_path)
    print(f"\nwarm-started v{from_version} -> saved v{to_version} at {save_path}.zip")
    print("evaluate with:  python3 controllers/ppo/ppo_log.py", to_version)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("from_version", help="source model version to continue from, e.g. 8")
    parser.add_argument("to_version", help="new version number to save as, e.g. 9")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=None)
    args = parser.parse_args()
    warmstart(args.from_version, args.to_version,
              total_timesteps=args.steps, n_envs=args.envs)
    
    # run with python3 controllers/ppo/ppo_warmstart.py _ _
    # first num version from, second num is to