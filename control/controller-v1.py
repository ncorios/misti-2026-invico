import numpy as np, mujoco, mujoco.viewer, os, time, matplotlib.pyplot as plt
import os, mujoco, mujoco.viewer, numpy as np
from control.pid import PID

# pid controller
import numpy as np
import matplotlib.pyplot as plt

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


# Intialize PID controllers for each joint (for simplicity, using same gains for all joints here)
kp = None
ki = None
kd = None
timestep = model.opt.timestep
pid_controllers = PID(kp, ki, kd, timestep)





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

        
       #  if is_walking: