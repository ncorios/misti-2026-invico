

import os, mujoco, mujoco.viewer, numpy as np, time, keyboard
from pid import PID   # your existing PID class

# ── Pausing ─────────────────────────────────────────────────────────────────────
Running = True
def _key_callback(key: int) -> None:
    global Running
    if key == 32:
        Running = not Running

# ── Model loading ──────────────────────────────────────────────────────────────
# Build an absolute path to the robot XML, one level up in the "plants" folder
xml_path = os.path.join(os.path.dirname(__file__), "..", "plants", "Prueba2_19_03.xml")
xml_path = os.path.abspath(xml_path)

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ── Initial pose ───────────────────────────────────────────────────────────────
# Find the keyframe named "stand" in the XML and reset the sim to that pose
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

# ── Resting joint angles (12 DOFs) ────────────────────────────────────────────
# Layout per leg: [abduction(0), hip(1), knee(2)]
# Legs: LF(0-2), RF(3-5), LH(6-8), RH(9-11)
desired_joint_angles = np.array([
    0.0,  0.65, -1.10,   # LF
    0.0,  0.65, -1.10,   # RF
    0.0,  0.65, -1.10,   # LH
    0.0,  0.65, -1.10,   # RH
], dtype=float)

# ── PID gains ─────────────────────────────────────────────────────────────────
KP = 15.0
KI = 0.1
KD = 0.2
TORQUE_LIMIT = 25.0   # Nm, matches XML ctrlrange="-25 25"

dt = model.opt.timestep  # 0.008 s

# One PID controller per joint; pre-seed theta so calc_D works from step 1.
initial_thetas = data.qpos[7:19].copy()   # joint angles after the 7-DOF free joint
pids = [PID(KP, KI, KD, dt) for _ in range(12)]
for i, pid in enumerate(pids):
    pid.update_theta(initial_thetas[i])

# ── Gait parameters ───────────────────────────────────────────────
gait_frequency       = 0.2
swing_amplitude      = 0.12
stance_amplitude     = 0.06
knee_amplitude       = 0.15
abduction_amplitude  = 0.07

# ── Timing ────────────────────────────────────────────────────────────────────
gait_time   = 0.0
TARGET_DIST = 2.0
debug_timer = 0.0

def phase_to_angle(sin_value):
    """Maps sine value → hip angle delta (same logic as original)."""
    if sin_value > 0:
        return swing_amplitude * sin_value
    else:
        return stance_amplitude * sin_value

# ── Main loop ──────────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1

    x_start = data.qpos[0]

    while viewer.is_running():
        if not Running:
            # Paused: just keep the viewer alive
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.004)
            continue

        x_current = data.qpos[0]
        y_current = data.qpos[1]
        distance_travelled = abs(x_current - x_start)
        is_walking = distance_travelled < TARGET_DIST

        # ── Debug print ───────────────────────────────────────────────────────
        debug_timer += 10 * dt
        if debug_timer >= 1.0:
            debug_timer = 0.0
            print(f"x={x_current:.4f}  y={y_current:.4f}  "
                  f"dist={distance_travelled:.4f}  gait_t={gait_time:.2f}")

        # ── Compute desired angles from gait oscillator ───────────────────────
        # Start from resting pose, then overlay the sine-wave deltas.
        desired = desired_joint_angles.copy()

        if is_walking:
            gait_phase = 2 * np.pi * gait_frequency * gait_time
            sin_A =  np.sin(gait_phase)
            sin_B =  np.sin(gait_phase + np.pi)

            # Hip joints (indices 1, 4, 7, 10)
            desired[1]  -= phase_to_angle(sin_A)   # LF hip
            desired[4]  -= phase_to_angle(sin_B)   # RF hip
            desired[7]  -= phase_to_angle(sin_B)   # LH hip
            desired[10] -= phase_to_angle(sin_A)   # RH hip

            # Knee joints (indices 2, 5, 8, 11)
            desired[2]  -= knee_amplitude * np.clip(sin_A, 0, 1)   # LF knee
            desired[5]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # RF knee
            desired[8]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # LH knee
            desired[11] -= knee_amplitude * np.clip(sin_A, 0, 1)   # RH knee

            # Abduction joints (indices 0, 3, 6, 9)
            desired[0]  += abduction_amplitude * np.clip(sin_A, 0, 1)   # LF abd
            desired[3]  += abduction_amplitude * np.clip(sin_B, 0, 1)   # RF abd
            desired[6]  -= abduction_amplitude * np.clip(sin_B, 0, 1)   # LH abd
            desired[9]  -= abduction_amplitude * np.clip(sin_A, 0, 1)   # RH abd

            gait_time += 10 * dt   # advance clock by the 10 physics steps below

        # ── PID torque computation & physics stepping ─────────────────────────
        # We run 10 physics steps per control cycle.
        # Inside each step: read current joint angle → update PID → write torque.
        for _ in range(10):
            current_thetas = data.qpos[7:19]   # live joint positions

            torques = np.zeros(12)
            for i, pid in enumerate(pids):
                pid.update_theta(current_thetas[i])        # feed back actual angle
                torques[i] = pid.calc_torque(desired[i])   # PID → torque

            # Clamp to actuator limits
            torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)

            data.ctrl[:] = torques
            mujoco.mj_step(model, data)

        viewer.sync()