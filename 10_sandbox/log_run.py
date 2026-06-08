"""
log_run.py — run the controller headless and save joint/desired/error arrays.

Drop this in the same folder as Testpid.py. It reuses your PID class and the
same gait + feedforward as Testpid (no viewer), then writes run_log.npz next
to this file. Then run plot_log.py to visualise.

    python log_run.py
"""
import os
import numpy as np
import mujoco
from pid import PID

HERE = os.path.dirname(os.path.abspath(__file__))
XML  = os.path.abspath(os.path.join(HERE, "..", "plants", "controler-v1.xml"))
OUT  = os.path.join(HERE, "run_log.npz")
T_SIM = 12.0   # seconds to simulate

# ── Model ──────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data,
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
mujoco.mj_forward(model, data)        # make qfrc_bias valid on the first step

# ── Setpoints / gains (same as Testpid) ─────────────────────────────────────────
desired_joint_angles = np.array([
    0.0, 0.65, -1.10,  0.0, 0.65, -1.10,
    0.0, 0.65, -1.10,  0.0, 0.65, -1.10,
], dtype=float)
KP, KI, KD = 10.0, 0.1, 0.5
TORQUE_LIMIT = 25.0
dt = model.opt.timestep

pid = PID(KP, KI, KD, dt)
pid.update_thetas(data.qpos[7:19].copy())

gait_frequency      = 0.1
swing_amplitude     = 0.18
stance_amplitude    = 0.09
knee_amplitude      = 0.22
abduction_amplitude = 0.07

def phase_to_angle(s):
    return swing_amplitude * s if s > 0 else stance_amplitude * s

# ── Feedforward state (init once) ───────────────────────────────────────────────
b            = model.dof_damping[6:18]
desired_prev = desired_joint_angles.copy()
gait_time    = 0.0
TARGET_DIST  = 2.0
x_start      = data.qpos[0]
n_cycles     = int(T_SIM / (10 * dt))

# ── Logs (one row per control cycle) ────────────────────────────────────────────
t_log, des_log, act_log, base_log, tau_log = [], [], [], [], []

for _ in range(n_cycles):
    is_walking = abs(data.qpos[0] - x_start) <= TARGET_DIST
    desired = desired_joint_angles.copy()

    if is_walking:
        gait_phase = 2 * np.pi * gait_frequency * gait_time
        sin_A, sin_B = np.sin(gait_phase), np.sin(gait_phase + np.pi)
        desired[1]  -= phase_to_angle(sin_A); desired[4]  -= phase_to_angle(sin_B)
        desired[7]  -= phase_to_angle(sin_B); desired[10] -= phase_to_angle(sin_A)
        desired[2]  -= knee_amplitude * np.clip(sin_A, 0, 1)
        desired[5]  -= knee_amplitude * np.clip(sin_B, 0, 1)
        desired[8]  -= knee_amplitude * np.clip(sin_B, 0, 1)
        desired[11] -= knee_amplitude * np.clip(sin_A, 0, 1)
        desired[0]  += abduction_amplitude * np.clip(sin_A, 0, 1)
        desired[3]  += abduction_amplitude * np.clip(sin_B, 0, 1)
        desired[6]  -= abduction_amplitude * np.clip(sin_B, 0, 1)
        desired[9]  -= abduction_amplitude * np.clip(sin_A, 0, 1)
        gait_time += 10 * dt

    qvel_des     = (desired - desired_prev) / (10 * dt)
    desired_prev = desired.copy()
    ff_v         = b * qvel_des

    # record at the start of the cycle
    t_log.append(gait_time)
    des_log.append(desired.copy())
    act_log.append(data.qpos[7:19].copy())
    base_log.append(data.qpos[0:7].copy())     # x y z + quaternion

    last_tau = np.zeros(12)
    for _ in range(10):
        current_thetas = data.qpos[7:19]
        ff = ff_v + data.qfrc_bias[6:18]
        pid.update_thetas(current_thetas)
        pid.update_errors(current_thetas, desired)
        last_tau = pid.calc_torque(ff, TORQUE_LIMIT)
        data.ctrl[:] = last_tau
        mujoco.mj_step(model, data)
    tau_log.append(last_tau.copy())

t    = np.array(t_log)
des  = np.array(des_log)
act  = np.array(act_log)
base = np.array(base_log)
tau  = np.array(tau_log)
err  = des - act

np.savez(OUT, t=t, desired=des, actual=act, error=err, base=base, torque=tau,
         kp=KP, ki=KI, kd=KD, dt=dt)

rms = np.sqrt(np.mean(err ** 2)) * 1e3
print(f"saved {OUT}")
print(f"base_z = {base[-1, 2]:.3f} m   overall RMS = {rms:.1f} mrad   "
      f"peak |err| = {np.abs(err).max()*1e3:.1f} mrad")
