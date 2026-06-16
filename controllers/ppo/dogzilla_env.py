import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "assets/ppo_dog.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# this comes from the AntEnv.

class DogEnv(MujocoEnv, utils.EzPickle):
    """
DOGZILLA S2 (XGO) quadruped locomotion environment.

A 12-DOF position-controlled quadruped (4 legs * abduction/hip/knee joints).

Adapted from the Gymnasium Ant-v5 MuJoCo environment, with observation space,
reward, and termination redefined for the real DOGZILLA S2 hardware so a trained
policy targets only quantities the physical robot can measure or command.

## Action Space
Box(-1.57, 1.57, (12,), float32). Each action is a target joint angle (radians)
sent to a position actuator (the XML uses <position> actuators, kp=35 for hip,
kp=55 for upper/lower). Actions map to joints in actuator order:
LF(hip, upper, lower), RF(hip, upper, lower), LH(hip, upper, lower), RH(hip, upper, lower).

This is position control, not torque control: the policy commands where each joint
should go, and MuJoCo's position actuator (mirroring the real servo's onboard loop)
drives it there. Range matches the joints' ctrlrange of [-1.57, 1.57].

## Observation Space
Box(-inf, inf, (22,), float64). Every element is something the real DOGZILLA can
measure at runtime — IMU + servo angle feedback only. No privileged simulator state.

| Index | Source            | Quantity                                            | Hardware sensor     |
|-------|-------------------|-----------------------------------------------------|---------------------|
| 0:3   | sensordata[0:3]   | gyroscope body-frame angular velocity (rad/s)       | IMU gyroscope       |
| 3:6   | sensordata[3:6]   | accelerometer body-frame proper accel incl. gravity | IMU accelerometer   |
| 6:10  | sensordata[6:10]  | base orientation quaternion (w,x,y,z)               | IMU (fused)         |
| 10:22 | qpos[7:19]        | 12 joint angles (rad)                               | servo angle sensors |

Deliberately excluded (available in sim, NOT on hardware — kept out of the policy
input to avoid a sim-to-real gap): base linear velocity (qvel[0:3]), per-joint
velocities, and contact forces. These may appear in the reward (sim-only at train
time) but never in the observation.

## Rewards
total = forward_reward + healthy_reward - smoothness_cost

- forward_reward: w_forward * base_x_velocity, where base_x_velocity is the x
  displacement of the `base` body over dt. Rewards walking forward.
- healthy_reward: constant per timestep while the robot is healthy (upright).
- smoothness_cost: w_smooth * sum((action - prev_action)^2). Penalizes jerky
  changes in joint targets, not action magnitude — magnitude penalties would
  punish the nonzero neutral stand pose (0, 0.65, -1.10 per leg).

Weights are tuned for this robot's scale (base height ~0.108 m, gram-scale links);
Ant's default weights do not transfer and are not used.

## Starting State
Reset to the `stand` keyframe pose (0, 0.65, -1.10 per leg) plus small uniform
noise, so each episode begins from the robot's actual standing configuration
rather than an all-zeros pose it never holds.

## Episode End
### Termination
The episode terminates when the robot is unhealthy: base height outside
[BASE_Z_MIN, BASE_Z_MAX] (a fallen dog's base drops near the floor), or any state
value non-finite. [Optionally also: roll/pitch beyond a tip-over threshold from the
framequat sensor.]

### Truncation
Episode length capped at [MAX_EPISODE_STEPS] timesteps (via TimeLimit wrapper or
registration).

## Reset
The robot is reset to the "stand" keyframe defined in the XML model:

<keyframe>
    <key name="stand"
         qpos="0 0 0.108 1 0 0 0   0 0.65 -1.10   0 0.65 -1.10   0 0.65 -1.10   0 0.65 -1.10"
         ctrl="0 0.65 -1.10   0 0.65 -1.10   0 0.65 -1.10   0 0.65 -1.10"/>
</keyframe>

qpos layout: base position (x, y, z = 0, 0, 0.108), base orientation quaternion
(w,x,y,z = 1,0,0,0, i.e. level), then 12 joint angles (hip, upper, lower) per leg
in order LF, RF, LH, RH — each leg holds (0, 0.65, -1.10) to stand.

qvel: linear torso velocites, joint angular velocities. 0 for standing keyframe, called anyways.

Adds noise (scaled by reset_noise_scale): uniform on joint positions, Gaussian
on velocities, so each episode starts slightly varied around the standing pose.
"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = DEFAULT_XML,
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 5.0,
        smoothness_cost_weight: float = 0.01,
        healthy_reward: float = 1.0,
        main_body: int | str = "base",
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.05, 0.17),
        reset_noise_scale: float = 0.00,
        upright_threshold: float = 0.5,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            smoothness_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            upright_threshold,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self.smoothness_cost_weight = smoothness_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._upright_threshold = upright_threshold
        self.previous_action = np.zeros(12)

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale


        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.sensordata.size + self.model.nu
        

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
           "sensordata": self.data.sensordata.size, # 10: gyro + accel + fusion
           "joint_angles": self.model.nu
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def smoothness_cost(self, action):
        smoothness_cost = self.smoothness_cost_weight * np.sum(np.square(action - self.previous_action))
        return smoothness_cost


    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        height_check = np.isfinite(state).all() and (min_z <= state[2] <= max_z)

        # orientation check: rotate body up-axis to world, see if it still points up
        quat = self.data.qpos[3:7]
        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), quat)
        orientation_check = up[2] > self._upright_threshold

        return height_check and orientation_check

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        self.previous_action = action
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        smoothness_cost = self.smoothness_cost(action)
        costs = smoothness_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_smoothness": -smoothness_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        """
        Build the 22-dim observation — only quantities the real DOGZILLA can measure.

        Layout (must match observation_space declared in __init__):
         [0:10]: sensordata  — IMU: gyro[3] + accelerometer[3] + orientation quat[4]
        [10:22]: qpos[7:19]  — 12 joint angles (servo readbacks)

        Excluded by design (privileged sim state, not on hardware):
        qpos[0:3]  base xyz position
        qpos[3:7]  base orientation (already have it via the quat sensor)
        qvel[*]    all velocities (base linear + joint angular)
        cfrc_ext   contact forces
        These may feed the reward (sim-only) but never the policy input.
        """

        sensor = self.data.sensordata          # 10: gyro + accel + quat
        joint_angles = self.data.qpos[7:19]    # 12 servo angles
        return np.concatenate([sensor, joint_angles])

    def reset_model(self):
        
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        # ── Initial pose ───────────────────────────────────────────────────────────────
        key = self.model.key("stand")
        key_qpos = key.qpos.copy()
        key_qvel = key.qvel.copy()
        qpos = key_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            key_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        self.previous_action = np.zeros(self.model.nu)

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }