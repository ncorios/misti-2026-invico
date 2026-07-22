"""
stand_test.py — plant sanity check (no policy).

Holds the robot at the `stand` keyframe defined in ppo_dog.xml, with the policy
removed, and checks it doesn't fall on its own over 200 steps. This isolates the
plant (model + actuators + standing pose) from the learned controller: if the dog
can't even stand still with no policy, the problem is the model/pose, not RL
exploration. This is the "remove the policy, see if the base system still works"
diagnostic that unblocked the residual-action-space decision early on.

The stand pose is read from the XML `stand` keyframe (qpos[7:19] = 12 joint
angles), not hardcoded, so it tracks the model if the keyframe ever changes.
DogEnv actions are RESIDUAL offsets added to that same stand pose
(joint_target = stand_pose + action), so the residual that reproduces the
keyframe exactly is all zeros.

Usage:
    python controllers/ppo/stand_test.py

Prints base height (qpos[2]) every 20 steps and reports whether the robot STOOD
for all 200 steps or FELL (termination fired) and when.
"""
import numpy as np
from dogzilla_env import DogEnv

env = DogEnv()  # headless, no render
obs, _ = env.reset()

# Stand pose from the XML `stand` keyframe — the pose DogEnv resets to and centers
# its residual action space on.
stand_pose = env.unwrapped.model.key("stand").qpos[7:19].copy()  # 12 joint angles
print(f"holding stand keyframe: {np.round(stand_pose, 3)}")

# Action is a residual around stand_pose, so a zero residual commands exactly the
# keyframe pose (no hardcoded joint angles).
hold_action = np.zeros(env.action_space.shape[0])

fell = False
for i in range(200):
    obs, r, term, trunc, info = env.step(hold_action)
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
