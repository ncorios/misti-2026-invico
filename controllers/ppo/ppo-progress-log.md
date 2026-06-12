# Progress Log for PPO - Nico

v-1 env approach
use established ant mujoco env, tweak to fit our specific needs.
what should the agent learn?
walk forward safely and quickly.
what information does the agent need?
 IMU orientation, IMU angular velocity, 12 joint angles. give the policy a few past timesteps for later implementations.
what actions can the agent take?
continous joint positions
how to measure success?
speed, smooth movement, reaching goal, not falling/failures
when should episodes end?
if it falls. if it reaches the "finish line", a defined distance it must travel.

rewards:
control_cost. keep. penalizes big actions
contact_cosst. reads cfrc_ext, contact forces. priviliged. not implementing in v_0.
is_healthy, fall check. ant checks if torso height is in range. this will be re written to torso height above x and roll/pitch within Y. 
__get_rew- computes reward. forward reward + healthy_reward - control cost- contact cost. rewrite.

terms:
step: doesnt change much
__get_obs. returns orientation, angular vel, 12 joint angles.
reset_model- reset to standing keyframe + small random noise for robustness.


step, reset_model, control_cost, _get_reset_info stay structurally intact. 
_get_obs, _get_rew, is_healthy, and the obs_size block in __init__ need rewrite