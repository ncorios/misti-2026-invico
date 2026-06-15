import os, mujoco, mujoco.viewer, numpy as np, time, keyboard
from pid import PID   # your existing PID class

# ── Pausing ─────────────────────────────────────────────────────────────────────
Running = True
def _key_callback(key: int) -> None:
    global Running
    if key == 32:
        Running = not Running

# ── Model loading ──────────────────────────────────────────────────────────────
xml_path = os.path.join(os.path.dirname(__file__), "controler-v1.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ── Initial pose ───────────────────────────────────────────────────────────────
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
KP = 10.0
KI = 0.1
KD = 0.5
TORQUE_LIMIT = 25.0   # Nm, matches XML ctrlrange="-25 25"
dt = model.opt.timestep  # 0.008 s

# One PID controller per joint; pre-seed theta so calc_D works from step 1.
initial_thetas = data.qpos[7:19].copy()
pid = PID(KP, KI, KD, dt)
pid.update_thetas(data.qpos[7:19].copy())

# ── Gait parameters ───────────────────────────────────────────────
gait_frequency       = .1
swing_amplitude      = 0.18
stance_amplitude     = 0.09
knee_amplitude       = 0.22
abduction_amplitude  = 0.07

# ── Timing ────────────────────────────────────────────────────────────────────
gait_time   = 0.0
TARGET_DIST = 2.0
debug_timer = 0.0

# ── Feedforward state (init ONCE, before the loop) ─────────────────────────────
b            = model.dof_damping[6:18]            # joint damping (= 2.0 per joint)
desired_prev = desired_joint_angles.copy()        # carries previous cycle's desired

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
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)
            continue
        time.sleep(.002)
        x_current = data.qpos[0]
        y_current = data.qpos[1]
        distance_travelled = abs(x_current - x_start)
        is_walking = distance_travelled <= TARGET_DIST

        # ── Debug print ───────────────────────────────────────────────────────
        debug_timer += 10 * dt
        if debug_timer >= 1.0:
            debug_timer = 0.0
            print(f"x={x_current:.4f}  y={y_current:.4f}  "
                  f"dist={distance_travelled:.4f}  gait_t={gait_time:.2f}")

        # ── Compute desired angles from gait oscillator ───────────────────────
        desired = desired_joint_angles.copy()

        if is_walking:
            gait_phase = 2 * np.pi * gait_frequency * gait_time
            sin_A =  np.sin(gait_phase)
            sin_B =  np.sin(gait_phase + np.pi)

            desired[1]  -= phase_to_angle(sin_A)   # LF hip
            desired[4]  -= phase_to_angle(sin_B)   # RF hip
            desired[7]  -= phase_to_angle(sin_B)   # LH hip
            desired[10] -= phase_to_angle(sin_A)   # RH hip

            desired[2]  -= knee_amplitude * np.clip(sin_A, 0, 1)   # LF knee
            desired[5]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # RF knee
            desired[8]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # LH knee
            desired[11] -= knee_amplitude * np.clip(sin_A, 0, 1)   # RH knee

            desired[0]  += abduction_amplitude * np.clip(sin_A, 0, 1)   # LF abd
            desired[3]  += abduction_amplitude * np.clip(sin_B, 0, 1)   # RF abd
            desired[6]  -= abduction_amplitude * np.clip(sin_B, 0, 1)   # LH abd
            desired[9]  -= abduction_amplitude * np.clip(sin_A, 0, 1)   # RH abd

            gait_time += 10 * dt

        # ── Feedforward: one finite-difference per control cycle ──────────────
        qvel_des     = (desired - desired_prev) / (10 * dt)   # desired joint velocity
        desired_prev = desired.copy()                          # carry to next cycle
        ff_v         = b * qvel_des                            # velocity feedforward

        # ── PID torque computation & physics stepping ─────────────────────────
        for _ in range(10):
            current_thetas = data.qpos[7:19]
            ff = ff_v + data.qfrc_bias[6:18]      # damping FF (per cycle) + gravity/bias (live)
            pid.update_thetas(current_thetas)
            pid.update_errors(current_thetas, desired)
            torques = pid.calc_torque(ff, TORQUE_LIMIT)
            data.ctrl[:] = torques
            mujoco.mj_step(model, data)

        print(data.qpos[2])
        viewer.sync()