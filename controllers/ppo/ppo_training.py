import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from dogzilla_env import DogEnv



class RewardTermsCallback(BaseCallback):
    """Logs reward_forward, reward_survive, reward_smoothness to TensorBoard."""
    
    def __init__(self):
        super().__init__()
        self._ep_reward_forward = []
        self._ep_reward_survive = []
        self._ep_reward_smoothness = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "reward_forward" in info:
                self._ep_reward_forward.append(info["reward_forward"])
                self._ep_reward_survive.append(info["reward_survive"])
                self._ep_reward_smoothness.append(info["reward_smoothness"])
        return True

    def _on_rollout_end(self) -> None:
        if self._ep_reward_forward:
            self.logger.record("reward/forward", 
                np.mean(self._ep_reward_forward))
            self.logger.record("reward/survive",  
                np.mean(self._ep_reward_survive))
            self.logger.record("reward/smoothness",
                np.mean(self._ep_reward_smoothness))
            self._ep_reward_forward.clear()
            self._ep_reward_survive.clear()
            self._ep_reward_smoothness.clear()
        
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def make_env():
    env = DogEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def train(version, total_timesteps=1_000_000, n_envs=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if n_envs is None:
        n_envs = max(1, os.cpu_count() - 2)

    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        verbose=1,
        tensorboard_log=os.path.join(HERE, "tb_logs"),
    )
    callback = RewardTermsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = os.path.join(MODEL_DIR, f"dog_ppo_v{version}")
    model.save(save_path)
    print(f"\nsaved model to {save_path}.zip")
    print("decide if its worth evaluating, then run python3 controllers/ppo/ppo_log.py", version)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version number for this run, e.g. 1")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=None)
    args = parser.parse_args()
    # SubprocVecEnv requires the __main__ guard — this is it
    train(args.version, total_timesteps=args.steps, n_envs=args.envs)

#copy and paste to run, add version, steps, envs at end: python3 controllers/ppo/ppo_training.py
