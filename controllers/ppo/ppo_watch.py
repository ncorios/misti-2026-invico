"""
ppo_watch.py — watch a trained PPO policy in the live MuJoCo viewer.

Loads models/dog_ppo_v{version}.zip, runs it in a DogEnv, and renders each step in a
passive viewer pinned to the XML "track" camera so it follows the dog. Resets and keeps
going when the episode ends, printing distance and whether it fell each episode. This is
the qualitative look at the gait; use ppo_log.py for the quantitative eval + recorded video.

Deterministic by default; pass --stochastic to sample actions (robustness check).

Run (macOS needs mjpython for the viewer):
    mjpython controllers/ppo/ppo_watch.py <version> [--stochastic]
    e.g. mjpython controllers/ppo/ppo_watch.py 41
"""
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from dogzilla_env import DogEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def watch(version, deterministic=True):
    model = PPO.load(os.path.join(MODEL_DIR, f"dog_ppo_v{version}"))
    env = DogEnv()
    obs, _ = env.reset()

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # point the viewer at the "track" camera from the XML so it follows the dog
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        ep = 0
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.unwrapped.dt)

            if terminated or truncated:
                ep += 1
                print(f"episode {ep}: fell={terminated}, x={info['x_position']:.2f}m")
                obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    watch(args.version, deterministic=not args.stochastic)
# run with mjpython controllers/ppo/ppo_watch.py <version>

# mjpython controllers/ppo/ppo_watch.py 41

