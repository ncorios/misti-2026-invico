import os, mujoco, mujoco.viewer , numpy as np

xml_path = os.path.join(os.path.dirname(__file__), "..", "plants", "Prueba2_19_03.xml")
xml_path = os.path.abspath(xml_path)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, key_id)

q_des = np.array([
    0.0,  0.65, -1.10,   # LF
    0.0,  0.65, -1.10,   # RF
    0.0,  0.65, -1.10,   # LH
    0.0,  0.65, -1.10,   # RH
], dtype=float)

freq       = 0.6
amp_swing  = 0.18
amp_stance = 0.09
amp_knee   = 0.22
amp_abd    = 0.07
dt           = model.opt.timestep
t            = 0.0
TARGET_DIST  = 100.0  # metros
debug_timer  = 0.0

def phase_signal(s):
    if s > 0:
        return amp_swing * s
    else:
        return amp_stance * s

with mujoco.viewer.launch_passive(model, data) as viewer:
    x_start = data.qpos[0]

    while viewer.is_running():
        x_current  = data.qpos[0]
        y_current  = data.qpos[1]
        distancia  = x_current - x_start
        caminando  = distancia < TARGET_DIST

        debug_timer += 10 * dt
        if debug_timer >= 1.0:
            debug_timer = 0.0
            print(f"x={x_current:.4f}  y={y_current:.4f}  dist={distancia:.4f}  t_marcha={t:.2f}")

        ctrl = q_des.copy()

        if caminando:
            phase = 2 * np.pi * freq * t
            s_A =  np.sin(phase)
            s_B =  np.sin(phase + np.pi)

            # Cadera
            ctrl[1]  -= phase_signal(s_A)   # LF
            ctrl[4]  -= phase_signal(s_B)   # RF
            ctrl[7]  -= phase_signal(s_B)   # LH
            ctrl[10] -= phase_signal(s_A)   # RH

            # Rodilla
            ctrl[2]  -= amp_knee * np.clip(s_A, 0, 1)   # LF
            ctrl[5]  -= amp_knee * np.clip(s_B, 0, 1)   # RF
            ctrl[8]  -= amp_knee * np.clip(s_B, 0, 1)   # LH
            ctrl[11] -= amp_knee * np.clip(s_A, 0, 1)   # RH

            # Abducción
            ctrl[0]  += amp_abd * np.clip(s_A, 0, 1)   # LF
            ctrl[3]  += amp_abd * np.clip(s_B, 0, 1)   # RF
            ctrl[6]  -= amp_abd * np.clip(s_B, 0, 1)   # LH
            ctrl[9]  -= amp_abd * np.clip(s_A, 0, 1)   # RH

            t += 10 * dt

        data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(model, data)
        viewer.sync()