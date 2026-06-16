import numpy as np
from dogzilla_env import DogEnv

env = DogEnv()  # no render, headless
obs, _ = env.reset()
stand_ctrl = np.array([0, 0.65, -1.10] * 4)

fell = False
for i in range(200):
    obs, r, term, trunc, info = env.step(stand_ctrl)
    z = env.unwrapped.data.qpos[2]
    if i % 20 == 0:
        print(f"step {i}: base_z={z:.3f}")
    if term or trunc:
        print(f">>> FELL at step {i}, base_z={z:.3f}")
        fell = True
        break
env.close()
if not fell:
    print(">>> STOOD for all 200 steps")