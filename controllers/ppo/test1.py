import gymnasium as gym
from stable_baselines3 import PPO

# train (headless, fast)
model = PPO("MlpPolicy", "Ant-v5", verbose=1)
model.learn(total_timesteps=100000)
model.save("ant_ppo")

# watch it (opens a window)
env = gym.make("Ant-v5", render_mode="human")
obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()