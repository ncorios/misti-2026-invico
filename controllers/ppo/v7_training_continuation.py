import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from dogzilla_env import DogEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def make_env():
    env = DogEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    return env


if __name__ == "__main__":
    n_envs = max(1, os.cpu_count() - 2)
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO.load(os.path.join(MODEL_DIR, "dog_ppo_v7"), env=vec_env)
    model.learn(total_timesteps=2_000_000, reset_num_timesteps=False)
    model.save(os.path.join(MODEL_DIR, "dog_ppo_v8"))
    print("saved dog_ppo_v8")