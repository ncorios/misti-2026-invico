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
forward_reward: rebalance crank weight up bc the mass and speed will be tiny so that it doesnt just farm healthy reward.

control_cost. keep. penalizes big actions. penalize action change tho instead of penalizing torque deviations from 0. so penalize np sum sum((action - prev_action)²)

contact_cost. reads cfrc_ext, contact forces. priviliged. is_healthy, fall check. ant checks if torso height is in range. this will be re written to torso height above x and roll/pitch within Y. dk if i will implement

is_healthy/termination. need new healthy z range, and incorporate an orientation check.

__get_rew- computes reward. forward reward + healthy_reward - control cost- contact cost. rewrite.

terms:
step: doesnt change much
__get_obs. returns orientation, angular vel, 12 joint angles.
reset_model- reset to standing keyframe + small random noise for robustness.


step, reset_model, control_cost, _get_reset_info stay structurally intact. 
_get_obs, _get_rew, is_healthy, and the obs_size block in __init__ need rewrite

obs:
qpos joint angles [7:19]
sensor data indexes for gyroscope, accelerometer, fusion

the robot has an imu with gyroscope and accelrometer. 6 axis. 
Gyroscope — 3-axis angular velocity (rad/s), in the sensor's body frame. The gyroscope measures rotational velocity along the X, Y and Z axes. 

Accelerometer — 3-axis proper acceleration (m/s²), including gravity. It senses static forces like gravity and dynamic forces like movement, over X, Y, and Z.
can get roll/pitch from fusion, yaw is unreliable bc no magnetometer correcting it
translating this to an updated xml. gives sesnordata of length 10: gyro[3] + accel[3] + quat[4]

 deployable _get_obs becomes: sensordata (10: gyro + accel + quat) + the 12 joint angles from qpos[7:19] = 22 numbers. 
 no qvel or contact forces bc privliged data. those will shape reward tho.

use stand for reset
obs size block redo
action box needs to match joing ctrl range from the XML. box +- 1.57. frame_skip/control frequency needs to match DOGZILLA servo loop. 
maybe wanna lower reset_noise_scale

non hardcoded actuator control range: self.action_space = Box(
    low=self.model.actuator_ctrlrange[:, 0],
    high=self.model.actuator_ctrlrange[:, 1],
    dtype=np.float32,
)

obs_size no hardcoding:
self.data.sensordata.size (10 from your three sensors) and joint count is self.model.nu
stand key frame: # in reset_model, instead of init_qpos:
stand_id = self.model.key("stand").id          # find the keyframe by name
qpos = self.model.key_qpos[stand_id].copy()    # its stored pose
qvel = np.zeros(self.model.nv)
qpos += self.np_random.uniform(low=-noise, high=noise, size=self.model.nq)
self.set_state(qpos, qvel)


later should write instructions for file path changing

to do for the week, get env working, run some runs, test evals and videos

for sim to real: 
clamp forcerange → yes, compute it from the 4.5 kg·cm spec (~0.44 N·m). Do this, it's a clear improvement and your current value is 50× off.
match kp/damping → not from a spec, but yes via system identification against the real servo's step response. That requires hardware access and a measurement, so it's a pre-deployment task, not a today task.
get weights and everything in xml as close to real robot as possible

masses,dynamics, inertia, etc are all approximate in the xml. fine for comparison/validation, need to ground for sim2real.

action space is prob too big rn. instant instability. switch to resiudal action space

BANG! v8 walks. goes in circles. next, penalize lateral yaw drift or reward moving in the forward facing direction.

i added yaw and stability costs. somehow made it worse lol. switched to a more robuste val bc the one episode eval before was making me think the model was bad when it wasnt.


