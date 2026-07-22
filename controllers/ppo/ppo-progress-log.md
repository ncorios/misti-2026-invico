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

## ok here i compiled some notes from other files

Distances are 10-episode means over 1000 steps, deterministic · stochastic, from
`ppo_eval/summary.csv`. Notes transcribed from each version's `vN-notes.md`.

### v1
_0.10 m / 0.0 surv · stoch 0.02 m / 0.0_
Why did it die: it fell over immediately, and i think the healthy z range was large enough
to not track this. i will add an orientation reward condition from qpos quat that will kill
episode.
current weights and z range: forward_reward_weight = 5.0, smoothness_cost_weight = 0.01,
healthy_reward = 1.0, main_body = "base", terminate_when_unhealthy = True,
healthy_z_range = (0.05, 0.17).

### v2
_0.11 m / 0.0 surv · stoch 0.04 m / 0.0_
all i did was increase forward reward weight to 20.
ts is lunging

### v5
_(no written notes.)_

### v6
_1.35 m / 0.0 surv · stoch 0.26 m / 0.0_
ok, checking if the initial pose is stable because the dog cant stay upright. wrote code to
check if it was stable in check stand.py, confirmed standing pose is fine. exploration and
action space is probably the problem then. instinct: "let me remove the policy and see if
the base system still works." keep standtest. make it usable for any keyframe.

### v7
_2.88 m / 0.6 surv · stoch 0.47 m / 0.1_
_(no written notes; the residual-action-space version v8 builds on.)_

### v8
_5.70 m / 0.3 surv · stoch 3.40 m / 0.1_
v8 built on v7, the residual. it now walks faster, and covered more distance and has almost
2x higher mean reward but died early. so faster but less stable. going to try warm start
from v8 and bumping up healthy reward from 1 to 2.

### v9
_6.42 m / 0.4 surv · stoch 5.41 m / 0.3_
hmm it got worse. going to make eval run multiple episodes and average to actually compare.
thinking about adding a stability term after comparing evals.
ok switched to means, but episodes are doing the same thing. might want to make deterministic
false or buff random noise at the beginning. std deviation 0 is strange.
ok switching to deterministic false + sampling episodes for v7-v9, i see that the policy is
much less robust than previously thought. v8 was better bc v9 had to adapt to a new reward.
going to run v10 on same healthy reward as v9 to see what happens.
im seeing healthy reward be 2 always. im thinking that it is too easy for the robot to hit.
probably need a stability reward first, then think about shrinking bounds on healthy
orientation and z range? after that a cost for yaw/drift.

### v10
_8.01 m / 0.6 surv · stoch 7.50 m / 0.5_
okay policy genuinely got better. going to probably focus on stability reward then walking
straight reward. ok i need my tensor board logs to be whatever they are for the version name
i give it, and to enable camera tracking in the video log. also add new rewards to reward
tracking for outputs during training.

### v11
_2.18 m / 0.4 surv · stoch 1.01 m / 0.0_
_(no written notes.)_

### v12
_6.43 m / 0.6 surv · stoch 2.84 m / 0.2_
okay policy got better. going to probably focus on stability reward then walking
straight reward. get tensorboard logs + camera tracking + new rewards in the output logs
working, then run some tests with different rewards disabled. i think need to increase
stability reward a lot more.
ok apparently doing single run was bad and unlucky and made me think that i needed more
rewards, when all i needed was a healthy bump. im gonna kill the new rewards and warm start
from v9, then work on reweighing the yaw punish reward.
starting ablation, yaw will be set to zero on 13.
yaw is structurally flawed so i will actually just keep it zero forever prob and do ablation
with stability 0 for 14 and then maybe healthy 0 for 15.

### v13
_0.50 m / 1.0 surv · stoch 0.12 m / 0.2_
ok back to circular motion lol. going to penalize y drift. y drift term when it turns 90
degrees can penalize any motion tho so if it stops moving after turning, turn down eight.
currently at .5. no circular motion without stability. likely with stability the easiest way
to get reward is some inefficient circular motion that still has positive x displacement.

### v14
_0.75 m / 0.3 surv · stoch 0.06 m / 0.1_
why the freak is he still spinning. time for ablation with stability 0. 

### v15
_0.41 m / 0.6 surv · stoch 0.14 m / 0.0_
ok its not rlly doing much. its just going back to y = 0 then messing up and correcting and
cycles. gonna try a warm start on v9 with new weights disappeared so just forward, healthy,
smoothness.
(i accidentally overwrote ablation model weights and notes)

### v16
_8.70 m / 0.5 surv · stoch 7.29 m / 0.5_
walks the best. drift and erratic bc theres nothing forcing joint symmetry. i wanna do a
joint symmetry term. doing a run w super small stability term just to see. .25 weight
stability, crank forward to 8?

### v17
_-0.02 m / 0.9 surv · stoch 0.12 m / 0.0_
okay this stability term is fried lol. idk what to do either something punishing asymmetrical
gait, force that in action space, limit speed maybe? or maybe think into exploring something
with the other gyro sensors like punishing tilt but that might be the same as our current
stability reward.
just one short asymmetry term. super low weight. will disable circular motions pretty much and
then you can add phase shifts later for yaw.

### v18
_-0.00 m / 0.3 surv · stoch 0.10 m / 0.2_
asymmetry term fully broke walking. dang. okay im gonna crank up stability to 1 and see what
happens (its a squared diff). all other rewards dead. prob gonna move to tuning
hyperparameters, as theres random scatter in the y drift. prob just not moving right. that
didnt work. gonna warm start v16 on way more timesteps now prob.

### v19
_11.28 m / 0.7 surv · stoch 8.45 m / 0.5_
ok great progress. tiny tiny y drift term incoming on warm start v19. will be in v20.

### v20
_13.87 m / 0.9 surv · stoch 13.67 m / 0.9_
okay y drift is helping for sure on bringing it back to center. stochastic got more brittle.
i think the step from here is train on 10 mil, then start doing domain randomization training
if gait is close to y = 0 enough. if not, crank up weight for y drift and retrain. its
probably time to move into hyperparameter tuning. im gonna read up on what all these numbers
mean, and what parameters to tune. ok dang so i accidentally rewrote the last model and didnt
warm start from it but yk its wtv.
honestly this is a great gait. amazing robustness good.
noise was 0.02, y drift at 0.04.
going to add a heading term, and run a fresh 5 mil step run.

### v21
_14.00 m / 0.8 surv · stoch 14.58 m / 1.0_
ok so the new heading reward doesnt punish when deviations get too far, so im gonna stack the
heading reward with the old one for pushes from both sides.

### v0 (heading-reward re-run; notes filed as v0/v21-notes.md)
_14.87 m / 0.9 surv · stoch 14.21 m / 0.8_
this is gonna be a v20 with a small heading reward on top and same y drift reward. heading
weight is .2. 10 mil timesteps. should basically just walk the same distance but get closer
to that y = 0 line. nevm it was 1 mil steps or something.
best run by far. push up heading and y drift and run it back.
deleted accidentally. 25 mil re run to compensate.

new heading cost code to punish at small deviations:

```python
@property
def heading_cost(self):
    """
    Penalize heading deviation from +x — the CAUSE of drift (walking at an angle),
    not the position symptom that y_drift chases.

    abs(yaw), not (1 - cos(yaw)): cosine's gradient vanishes near zero, so small
    headings feel no pressure and drift stalls (~1m). abs(yaw) keeps constant pull
    even at tiny errors, driving residual drift to zero.

    yaw ANGLE, not rate: rate locks in bad headings (the turning_cost mistake);
    angle-from-+x creates a restoring pull that allows correction.

    yaw = atan2(forward_y, forward_x), where forward = body [1,0,0] rotated to world.
    0 = facing +x (no cost), +/-pi = facing -x. Raising weight tightens drift but
    fights robustness — watch survival_rate. Verify cost ~0 at stand pose.
    """
    quat = self.data.qpos[3:7]
    forward = np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([1.0, 0.0, 0.0]), quat)
    yaw = np.arctan2(forward[1], forward[0])
    return self.heading_cost_weight * abs(yaw)
```

old code:

```python
def heading_cost(self):
    # penalize the body pointing away from +x (the cause of drift: walking at an angle).
    # rotate the body's forward axis [1,0,0] into the world by the base quaternion,
    # then reward its x-component being ~1 (nosed straight down +x).
    # penalize (1 - forward_x): 0 when facing +x, grows to 2 when facing -x.
    # note: yaw ANGLE (deviation from +x), NOT yaw rate — rate locks in bad headings,
    # angle creates a restoring pull toward facing forward (allows correction).
    quat = self.data.qpos[3:7]
    forward = np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([1.0, 0.0, 0.0]), quat)
    deviation = 1.0 - forward[0]      # 0 when facing +x, up to 2 when facing -x
    return self.heading_cost_weight * deviation
```

i accidentally deleted this model.

heading new code (stacked for both limits):

```python
@property
def heading_cost(self):
    """
    Penalize heading deviation from +x — the CAUSE of drift (walking at an angle),
    not the position symptom that y_drift chases.

    abs(yaw), not (1 - cos(yaw)): cosine's gradient vanishes near zero, so small
    headings feel no pressure and drift stalls (~1m). abs(yaw) keeps constant pull
    even at tiny errors, driving residual drift to zero.

    yaw ANGLE, not rate: rate locks in bad headings (the turning_cost mistake);
    angle-from-+x creates a restoring pull that allows correction.

    yaw = atan2(forward_y, forward_x), where forward = body [1,0,0] rotated to world.
    0 = facing +x (no cost), +/-pi = facing -x. Raising weight tightens drift but
    fights robustness — watch survival_rate. Verify cost ~0 at stand pose.

    STACKED FOR BOTH LIMITS
    """
    quat = self.data.qpos[3:7]
    forward = np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([1.0, 0.0, 0.0]), quat)
    yaw = np.arctan2(forward[1], forward[0])   # signed heading error from +x
    # |yaw|: steady pull even at small errors (keeps it tight near zero)
    # yaw**2: escalating pull, yanks back hard when far (fixes "leaves when far")
    return self.heading_lin_weight * abs(yaw) + self.heading_quad_weight * yaw**2
```

### v22
_7.92 m / 0.4 surv · stoch 9.94 m / 0.4_
fresh 10 mil step run, 0.04 y drift, 0.03 heading, 0.02 noise. didnt rlly get much better.

### v23
_0.79 m / 0.0 surv · stoch 0.43 m / 0.0_
_(no written notes; collapsed to standing-still from an over-escalated heading cost.)_

### v24
_15.96 m / 0.9 surv · stoch 15.22 m / 0.9_
beautiful gait, getting much better. going to crank up y drift reward and then do a run
cranking stability and heading.
results — new best model. energy penalty (w=0.0005, ~3% of forward) was a clean win on every
axis:
- v0 (old best): 14.30m det / 0.70 surv, 13.92m stoch / 0.90
- v24: 15.30m det / 0.90 surv, 15.56m stoch / 0.90
energy term cleaned the gait AND improved distance + survival. it's also the in-sim
cost-of-transport proxy for the comparison study.
known issue — galloping. v24 walks with a bound/gallop — a pitch oscillation, not a sustained
lean. measured over 1000 det steps: tilt (1-up_z) mean 0.019 / peak 0.106 (~11deg mean,
~27deg peak) = modest; pitch RATE (gyro-y) mean 2.74 rad/s (~157 deg/s) / peak 10.1
(~577 deg/s) = large. energy penalties tend to push toward a ballistic bound because it's
cheap at speed. doesn't hurt sim metrics, but it's off-target for "normal quadruped walking"
and would be violent on real hardware (sim-to-real).
FUTURE IMPLEMENTATION: anti-gallop leveling term (do NOT use a pitch-RATE cost). Add a
leveling term to damp the gallop. Use an ANGLE-based cost, NOT a rate-based one.
WHY NOT pitch-rate (w*|gyro_y|): it repeats the turning_cost mistake. Penalizing a rate
punishes ALL pitching, including the corrective/recovery pitch needed to catch a stumble.
It "works" only by making the body stiffer (less able to recover) — watch stochastic survival
drop. A rate cost cannot allow corrections, same reason turning_cost failed.
WHY angle (leveling): penalize deviation-from-level, not the motion. Returning to level
REDUCES cost -> a restoring pull toward upright that still allows recovery pitching. Same
principle as the heading-ANGLE fix that replaced turning_cost.
CATCH: this gallop is mostly visible in rate (mean tilt only ~0.019), and plain (1-up_z) has
a vanishing gradient near level, so a small leveling weight may be too weak and a large one
forces a rigidly flat posture. FIX (reuse the heading lin+quad trick): a lin+quad PITCH-ANGLE
cost so it bites even at small tilt while staying angle-based:

```python
pitch = arcsin(clip(forward_z, -1, 1))   # forward = body [1,0,0] rotated to world
cost  = pitch_lin_weight * abs(pitch) + pitch_quad_weight * pitch**2
```

(pitch-specific, so it won't fight roll needed for turning/balance. start small, watch
forward progress + stochastic survival — leveling, like heading, can freeze walking if
overweighted, as v23 showed.)

### v25
_14.47 m / 0.8 surv · stoch 14.76 m / 0.9_
_(no written notes.)_

### v26
_15.83 m / 0.8 surv · stoch 13.61 m / 0.7_
warmstart from v25. basically cranked down energy term, cranked up. finally tweaking
hyperparameters, gonna raise y drift and heading a tiny bit and re run. heading cost the same,
y drift cranked up from 0.04 to 0.05. basically learning rate is getting lowered from 0.003 to
0.001. this will reduce flip and make training more stable. std is a bit flat, so going to
lower ent_coef. explained variance is pretty noisy so will lower learning rate as well. warm
starting with these changes into v27, hopefully will be the one. honestly if the error is
within +-0.5m on the y im fine with that, bc producing a straight gait is very hard without
amp or outputting only half of the joint angles and adding a sinusoidal transformation for the
other joint angles. this was the breakthrough, learning rate + just keep bumping the weights to
get it to center more and warm starting. idc about stochastic being rough, ill do some domain
randomization later, rn i want a tight deterministic policy.

### v27
_17.68 m / 1.0 surv · stoch 13.54 m / 0.7_
_(no written notes; the hyperparameter-tuning breakthrough warmstart described in v26.)_

### v28
_16.30 m / 0.9 surv · stoch 16.40 m / 0.8_
_(no written notes.)_

### v29
_16.69 m / 0.9 surv · stoch 12.95 m / 0.6_
basically this guy gallops really fast and really far. gonna do like a 25m timestep run with
the new hyperparameters and new weights to try to get some stable gait. honestly im realizing
that the generation of a gallop vs the generation of a trot is super different. i should have
had a term forcing diagonal pairs with a phase shift. paths are either contact force on opposing feet with
sensors added to the xml, or an asymmetry term but with like a phase shift and defined period
(theta_1(t) = theta_2(t + T/2)) for example.
old weights were around 0.005 for energy, heading cost 0.2, y drift was around 0.08,
heading_center_gain was around 1. weights:
```
forward_reward_weight = 5.0, smoothness_cost_weight = .01, healthy_reward_weight = 2,
stability_reward_weight = 0, turning_cost_weight = 0, y_drift_cost_weight = 0.1,
asymmetry_cost_weight = 0, heading_cost_weight = 0.4, heading_lin_weight = 0.1,
heading_quad_weight = 0.005, heading_center_gain = 3.0, heading_max_correction = 0.05,
energy_cost_weight = 0.008, terminate_when_unhealthy = True,
healthy_z_range = (0.05, 0.17), reset_noise_scale = 0.02, upright_threshold = 0.5
```
hyperparameters: learning_rate = 3e-5, lr_schedule = lambda _: 3e-5, ent_coef = 0.0,
target_kl = 0.0175. v30 will be launched from v29 with 3 mil steps to see if these weights
break gait.

### v30
_16.24 m / 0.8 surv · stoch 14.14 m / 0.6_
_(no written notes; the 3 mil step launch from v29 to test whether the new weights break the
gait.)_

### v31
_16.84 m / 0.9 surv · stoch 16.68 m / 0.9_
im not sure why, but theres like an implicit bias whenever i crank y drift. it kinda flip
flops and picks a direction to go to. v32 decrease forward to 4, make heading and y drift 1
cause why not. also this is warm starting from v26 bc it was the straightest.

### v32
_13.73 m / 0.7 surv · stoch 14.19 m / 0.9_
_(no written notes; the v31→v32 change: forward down to 4, heading and y drift to 1.)_

### v33
_18.63 m / 1.0 surv · stoch 15.38 m / 0.8_
_(no written notes.)_

### v35
_1.85 m / 0.3 surv · stoch 1.72 m / 0.1_
_(no written notes.)_

### v41
_18.04 m / 0.97 surv · stoch 16.13 m / 0.84_
the lack of notes is bc i did a very long run of warm starts, culminating in this version. gaussian noise causing random drift, noiseless walks straight.
i basically decreased forward reward to 2.5 and raised heading and y drift to 1. bang.
should get a no noise video. need [notes end here].


