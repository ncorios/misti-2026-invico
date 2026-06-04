import os, mujoco, mujoco.viewer, numpy as np

# ── Model loading ──────────────────────────────────────────────────────────────
# Build an absolute path to the robot XML, one level up in the "plants" folder
xml_path = os.path.join(os.path.dirname(__file__), "..", "plants", "Prueba2_19_03.xml")
xml_path = os.path.abspath(xml_path)

# Load the MuJoCo model and create a simulation data object
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# ── Initial pose ───────────────────────────────────────────────────────────────
# Find the keyframe named "stand" in the XML and reset the sim to that pose
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

# Desired (resting) joint angles for all 12 actuated DOFs.
# Layout per leg: [abduction, hip, knee]
# Legs: LF = Left Front, RF = Right Front, LH = Left Hind, RH = Right Hind
desired_joint_angles = np.array([
    0.0,  0.65, -1.10,   # LF
    0.0,  0.65, -1.10,   # RF
    0.0,  0.65, -1.10,   # LH
    0.0,  0.65, -1.10,   # RH
], dtype=float)

# ── Gait parameters ────────────────────────────────────────────────────────────
gait_frequency  = 0.6    # cycles per second (Hz)
swing_amplitude = 0.18   # hip angle added while the leg is in swing phase (s > 0)
stance_amplitude= 0.09   # hip angle added while the leg is in stance phase (s < 0)
knee_amplitude  = 0.22   # knee retraction during swing
abduction_amplitude = 0.07  # lateral abduction during swing

# ── Timing & stopping condition ────────────────────────────────────────────────
timestep     = model.opt.timestep  # simulation timestep (seconds), from XML
gait_time    = 0.0                 # accumulated gait clock (only advances while walking)
TARGET_DIST  = 100.0               # stop gait oscillation after travelling this far (metres)
debug_timer  = 0.0                 # accumulator used to throttle console prints

# ── Phase-to-angle helper ──────────────────────────────────────────────────────
def phase_to_angle(sin_value):
    """
    Maps a sine value to a hip joint delta.
    Positive  → swing phase  → larger amplitude (leg lifting forward).
    Negative  → stance phase → smaller amplitude (leg pushing back).
    """
    if sin_value > 0:
        return swing_amplitude * sin_value
    else:
        return stance_amplitude * sin_value

# ── Main simulation loop ───────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    # lock camera onto body 1 (usually the torso/base link)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1  # change to the body index of your robot's base
    
    x_start = data.qpos[0]
    while viewer.is_running():
        

        # Current robot position in world frame
        x_current = data.qpos[0]
        y_current = data.qpos[1]
        distance_travelled = x_current - x_start

        # Walk until we've covered TARGET_DIST; after that hold the resting pose
        is_walking = distance_travelled < TARGET_DIST

        # ── Debug print (≈ every 1 simulated second) ──────────────────────────
        debug_timer += 10 * timestep          # 10 steps are taken per control loop
        if debug_timer >= 1.0:
            debug_timer = 0.0
            print(f"x={x_current:.4f}  y={y_current:.4f}  "
                  f"dist={distance_travelled:.4f}  gait_t={gait_time:.2f}")

        # Start from the desired resting angles each control cycle
        ctrl = desired_joint_angles.copy()

        if is_walking:
            # ── Gait oscillator ───────────────────────────────────────────────
            # Two sine waves 180° out of phase drive a trotting diagonal gait:
            #   phase_A → LF + RH (diagonal pair A)
            #   phase_B → RF + LH (diagonal pair B)
            gait_phase = 2 * np.pi * gait_frequency * gait_time
            sin_A =  np.sin(gait_phase)           # diagonal pair A
            sin_B =  np.sin(gait_phase + np.pi)   # diagonal pair B (anti-phase)

            # ── Hip joints (indices 1, 4, 7, 10) ─────────────────────────────
            # Subtract so that a positive phase_to_angle swings the leg forward
            ctrl[1]  -= phase_to_angle(sin_A)   # LF hip
            ctrl[4]  -= phase_to_angle(sin_B)   # RF hip
            ctrl[7]  -= phase_to_angle(sin_B)   # LH hip
            ctrl[10] -= phase_to_angle(sin_A)   # RH hip

            # ── Knee joints (indices 2, 5, 8, 11) ────────────────────────────
            # Retract (flex) the knee only during swing (positive half of sine)
            ctrl[2]  -= knee_amplitude * np.clip(sin_A, 0, 1)   # LF knee
            ctrl[5]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # RF knee
            ctrl[8]  -= knee_amplitude * np.clip(sin_B, 0, 1)   # LH knee
            ctrl[11] -= knee_amplitude * np.clip(sin_A, 0, 1)   # RH knee

            # ── Abduction joints (indices 0, 3, 6, 9) ─────────────────────────
            # Front legs abduct outward (+), hind legs abduct inward (−) during swing
            ctrl[0]  += abduction_amplitude * np.clip(sin_A, 0, 1)   # LF abd
            ctrl[3]  += abduction_amplitude * np.clip(sin_B, 0, 1)   # RF abd
            ctrl[6]  -= abduction_amplitude * np.clip(sin_B, 0, 1)   # LH abd
            ctrl[9]  -= abduction_amplitude * np.clip(sin_A, 0, 1)   # RH abd

            # Advance the gait clock by 10 timesteps (matching the step loop below)
            gait_time += 10 * timestep

        # ── Apply controls and step the physics ───────────────────────────────
        data.ctrl[:] = ctrl
        for _ in range(10):           # integrate 10 physics steps per control cycle
            mujoco.mj_step(model, data)
        viewer.sync()                 # push updated state to the passive viewer