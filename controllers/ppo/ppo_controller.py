import numpy as np
import os
from stable_baselines3 import PPO
from dogzilla_env import DogEnv

class PPOController:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
    def __call__(self, command, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action   # already np.ndarray(12), already your joint targets """

# load_path = os.path.join(MODEL_DIR, f"dog_ppo_v{from_version}")
# model = PPO.load(load_path, env=vec_env)