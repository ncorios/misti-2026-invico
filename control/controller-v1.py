import numpy as np, mujoco, mujoco.viewer, os, time, matplotlib.pyplot as plt
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
