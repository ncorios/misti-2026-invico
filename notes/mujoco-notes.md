# MuJoCo Reference: XML + Python

Everything here is explained, everything explained is used in the example at the bottom.

---
0
## Part 1: The XML Model

A MuJoCo model is a single `.xml` file (called MJCF) that describes the full physical world: bodies, joints, geometry, actuators, and sensors. MuJoCo reads this file, compiles it into a physics model, and simulates it. You never write physics equations — you describe the system, MuJoCo does the math.

The file has a fixed set of top-level sections inside a `<mujoco>` root element. Here's every section you need, in order.

### `<compiler>`

Controls how MuJoCo *parses* the XML. Not physics — just interpretation rules.

```xml
<compiler angle="radian" inertiafromgeom="true"/>
```

**`angle`** — `"radian"` or `"degree"`. Determines how MuJoCo reads every angle value in the file: joint ranges, axis-angle rotations, everything. Default is `"degree"`. Set this to `"radian"` and forget about it — radians are what your controller works in, what `data.qpos` returns, and what every dynamics equation expects. If you leave this as degree and your PID outputs radians, you'll get garbage and it'll be invisible.

**`inertiafromgeom`** — if `"true"`, MuJoCo automatically computes each body's mass and rotational inertia from its geometry shapes and their densities. If `"false"` (the default), you have to manually specify an `<inertial>` block on every body with mass, center of mass, and inertia tensor — tedious and error-prone. Set this to `"true"` for any prototype work.

### `<option>`

Global physics parameters that apply to the entire simulation.

```xml
<option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
```

**`timestep`** — seconds per simulation step. Every call to `mj_step()` advances time by this much. Smaller = more accurate but slower. `0.002` (2 ms) is a safe default. If your simulation explodes or contacts look wrong, try `0.001`.

**`gravity`** — a 3D vector `"x y z"` in m/s². `"0 0 -9.81"` means gravity pulls down along the Z axis. MuJoCo uses Z-up by convention. Set to `"0 0 0"` to test your controller without gravity (useful for isolating the P-term response).

**`integrator`** — the numerical method MuJoCo uses to advance the state. Options:
- `"Euler"` — simplest, fastest. Fine for most control work.
- `"RK4"` — 4th-order Runge-Kutta. More accurate per step, ~4× more computation. Good default when you care about accuracy.
- `"implicit"` / `"implicitfast"` — for stiff systems (springs, soft contacts). You won't need these for a rigid arm.

### `<worldbody>`

The scene tree. This is where all physical objects live. It contains nested `<body>` elements, each of which can contain geometry, joints, sensors, and child bodies.

MuJoCo's coordinate system is **right-handed, Z-up**. All vectors are written as `"x y z"` with spaces.

#### `<light>`

A light source for the viewer. Not physics — just so you can see what's happening.

```xml
<light pos="0 0 3" dir="0 0 -1"/>
```

`pos` is where the light is, `dir` is what direction it points.

#### `<geom>` (at the world level)

A geometry shape that belongs to the world (the ground, walls, etc.). World-level geoms are static — they don't move.

```xml
<geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
```

**`type`** — the shape. Common types:
- `"plane"` — infinite flat surface. `size` controls visual extent only.
- `"box"` — rectangular solid. `size="x y z"` gives *half-extents* (so `size="0.05 0.05 0.05"` makes a 10cm cube).
- `"capsule"` — cylinder with hemispherical endcaps. Defined by `size="radius"` and either `fromto="x1 y1 z1  x2 y2 z2"` (two endpoints) or `size="radius half_length"`.
- `"sphere"` — `size="radius"`.
- `"cylinder"` — `size="radius half_length"`.
- `"mesh"` — uses a mesh from `<asset>`. For imported STL/OBJ files.

**`rgba`** — color as `"r g b a"`, each 0 to 1. `a` is opacity.

**`mass`** — mass in kg. When `inertiafromgeom="true"`, MuJoCo uses this (or `density`) to compute the body's inertia. If you set `mass`, it overrides `density` for that geom.

**`fromto`** — defines a capsule (or cylinder) by its two endpoints: `"x1 y1 z1  x2 y2 z2"`. The shape stretches between these two points. Way more intuitive than `pos` + `size` for elongated shapes like arm links.

#### `<body>`

A rigid body. Has a position and orientation relative to its parent. Can contain joints, geoms, sites, and child bodies.

```xml
<body name="base" pos="0 0 0.5">
```

**`name`** — string identifier. You'll use this to look up the body in Python.

**`pos`** — position `"x y z"` relative to the parent body (or world, if this is a top-level body). This is where the body's origin sits when all joints are at zero.

Bodies form a tree. Nesting a `<body>` inside another creates a parent-child kinematic chain. The child's `pos` is measured in the parent's frame. When the parent moves, the child moves with it.

#### `<joint>`

A degree of freedom between a body and its parent. Without a joint, a body is welded rigidly to its parent.

```xml
<joint name="elbow" type="hinge" axis="0 1 0"
       range="-1.57 1.57" damping="0.1" armature="0.01"
       limited="true"/>
```

**`type`** — what kind of motion:
- `"hinge"` — rotation around a single axis. 1 DOF. This is what you want for a revolute joint.
- `"slide"` — translation along a single axis. 1 DOF.
- `"ball"` — rotation around all three axes. 3 DOF.
- `"free"` — full 6-DOF (3 translation + 3 rotation). For floating bodies like a quadrotor.

**`axis`** — the axis of rotation (for hinge) or translation (for slide), as a unit vector `"x y z"` in the body's frame. `"0 1 0"` means rotation around the local Y axis — the arm swings in the XZ plane.

**`range`** — joint limits as `"min max"`. In radians (because we set `angle="radian"` in compiler). `"-1.57 1.57"` is ±90°. Only enforced if `limited="true"`.

**`limited`** — `"true"` or `"false"`. Must be `"true"` for `range` to do anything. (You can also set `autolimits="true"` in `<compiler>` to skip this — any joint with a `range` automatically becomes limited.)

**`damping`** — velocity-proportional resistance at the joint, in N·m·s/rad. Applies a torque of `-damping * qdot` at every timestep. This is passive physical damping (like friction or viscosity in the joint), not your D-term. A small value (0.1) keeps things stable without dominating dynamics. Too high and the joint moves like it's underwater.

**`armature`** — reflected motor inertia, in kg·m². Real motors have rotational inertia that gets reflected through gear ratios. Adding a small value (0.01) makes the joint behave more like a real actuated joint and also helps numerical stability. Without it, a massless joint can cause simulation instabilities.

#### `<site>`

A named point fixed to a body. No mass, no collision — just a reference marker.

```xml
<site name="tip" pos="0 0 0.3" size="0.01"/>
```

`pos` is relative to the body it's inside. You can read a site's world-frame position in Python with `data.site_xpos`. Use sites for end-effector tracking, sensor attachment points, or camera targets. `size` just controls the visual dot size in the viewer.

### `<actuator>`

How you apply forces to joints. Each actuator creates one slot in the `data.ctrl` array.

```xml
<motor name="elbow_motor" joint="elbow" gear="10"
       ctrllimited="true" ctrlrange="-10 10"/>
```

**`<motor>`** — applies raw torque to a joint. The torque is `ctrl * gear`. This is the actuator type you want for PID control — you compute the torque, you set `data.ctrl[0]`, MuJoCo applies it.

**`gear`** — scaling factor. If your PID outputs values in [-1, 1], set `gear` to your max torque (e.g., 10 N·m). Then `ctrl=1.0` applies 10 N·m. Alternatively, set `gear="1"` and have your PID output actual torques.

**`ctrllimited`** / **`ctrlrange`** — clamps the control signal before applying it. `ctrlrange="-10 10"` means MuJoCo will clip anything outside [-10, 10]. This is actuator saturation — the same thing that triggers integral windup in your PID. With this set, you know your anti-windup logic is being tested realistically.

Other actuator types you should *not* use for this project:
- `<position>` — has a built-in P controller inside MuJoCo. It will fight your PID or make it redundant.
- `<velocity>` — has a built-in velocity controller. Same problem.

Both are useful for quick prototyping when you don't want to write a controller, but they defeat the purpose of what you're building.

### `<sensor>`

Measurements you can read from `data.sensordata` in Python. MuJoCo stacks sensor values in the order they're declared.

```xml
<sensor>
  <jointpos name="q"    joint="elbow"/>
  <jointvel name="qdot" joint="elbow"/>
</sensor>
```

**`<jointpos>`** — reads the joint angle in radians. For a hinge joint, this is a single scalar.

**`<jointvel>`** — reads the joint angular velocity in rad/s.

Other sensor types you'll use later:
- `<actuatorfrc>` — the torque actually applied by an actuator.
- `<framepos>` — 3D world position of a site (for end-effector tracking).
- `<touch>` — contact normal force at a site.

### `<keyframe>`

Named initial states. A keyframe stores joint positions (and optionally velocities and controls) that you can reset to.

```xml
<keyframe>
  <key name="home" qpos="0"/>
</keyframe>
```

**`qpos`** — joint positions for every joint, in declaration order. For a single hinge, just one number. For a 4-DOF robot, four numbers separated by spaces.

In Python, reset to a keyframe with:
```python
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
```

Way cleaner than manually setting `data.qpos[0] = 0; data.qvel[0] = 0` every time.

---

## Part 2: Python API

### Loading and stepping

```python
import mujoco

model = mujoco.MjModel.from_xml_path("plants/arm.xml")   # compile XML → model
data  = mujoco.MjData(model)                               # create simulation state
```

`model` is the static description (masses, geometry, joint structure). It never changes.
`data` is the live state (positions, velocities, forces). It changes every step.

```python
mujoco.mj_step(model, data)     # advance simulation by one timestep
```

This is the core call. It integrates the equations of motion: applies gravity, contact forces, actuator forces, joint damping — everything specified in the XML — and updates `data`.

### Reading state

```python
data.qpos[0]    # joint position (rad) — ground truth from the physics engine
data.qvel[0]    # joint velocity (rad/s) — ground truth
data.time       # current simulation time (seconds)
```

`qpos` and `qvel` are indexed by joint order (declaration order in the XML). For a single-joint model, index 0 is your joint.

### Reading sensors

```python
data.sensordata[0]    # first sensor value (jointpos "q")
data.sensordata[1]    # second sensor value (jointvel "qdot")
```

Sensors are stacked in declaration order. You can also look up by name:

```python
idx = model.sensor("q").id
data.sensordata[idx]
```

For this project, `data.sensordata` and `data.qpos`/`data.qvel` return the same values. The difference matters later: `sensordata` is where you'd inject noise or delays to simulate real hardware. `qpos`/`qvel` is always the exact truth.

### Writing control

```python
data.ctrl[0] = torque_value    # set actuator 0's control signal
```

One entry per actuator, in declaration order. MuJoCo multiplies by `gear`, clamps by `ctrlrange`, and applies the resulting torque at the next `mj_step`.

### Reading body/site positions

```python
data.xpos[model.body("forearm").id]      # body's world-frame position (3D)
data.site_xpos[model.site("tip").id]     # site's world-frame position (3D)
```

### Resetting

```python
mujoco.mj_resetData(model, data)         # reset everything to XML defaults
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)  # reset to keyframe
```

### Rendering (passive viewer)

```python
import mujoco.viewer

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

`launch_passive` opens a window. `viewer.sync()` pushes the current state to the display. The window is interactive — you can rotate, zoom, and pause.

### Getting the timestep

```python
dt = model.opt.timestep    # whatever you set in <option timestep="..."/>
```

You'll pass this to your PID's `update()` method.

---

## Part 3: Complete Working Example

### `plants/arm.xml`

```xml
<mujoco model="1dof_arm">

  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>

    <body name="base" pos="0 0 0.5">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>

      <body name="forearm" pos="0 0 0.05">
        <joint name="elbow" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.1" armature="0.01"
               limited="true"/>
        <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.3"
              rgba="0.7 0.3 0.2 1" mass="0.5"/>
        <site name="tip" pos="0 0 0.3" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="elbow_motor" joint="elbow" gear="10"
           ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>

  <sensor>
    <jointpos name="q"    joint="elbow"/>
    <jointvel name="qdot" joint="elbow"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0"/>
  </keyframe>

</mujoco>
```

**What this builds:** A fixed base (gray cube) mounted 0.5m above the floor. A forearm (orange capsule, 0.3m long, 0.5 kg) hangs from the base, connected by a hinge joint that rotates around the Y axis. A motor applies up to ±100 N·m of torque (ctrl range [-10, 10] × gear 10). Two sensors report joint angle and velocity. Gravity pulls the forearm down. Your PID's job is to hold it at a target angle.

### `mujoco_experiments/sim.py`

```python
"""
1-DOF arm simulation with PID control.
Reads theta and theta_dot from sensors, applies torque through motor.
Logs data and plots step response.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ── simple PID (standalone, no imports from control/) ──────────────

class PID:
    def __init__(self, kp, ki, kd, ctrl_min=-10, ctrl_max=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        # proportional
        p = self.kp * error

        # integral with clamping anti-windup
        self.integral += error * dt
        i = self.ki * self.integral

        # derivative (of error, not measurement — fine for step reference)
        if self.prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        # total output
        output = p + i + d

        # clamp and apply anti-windup
        clamped = np.clip(output, self.ctrl_min, self.ctrl_max)
        if clamped != output:
            # undo the integral accumulation that caused saturation
            self.integral -= error * dt

        return clamped


# ── simulation ─────────────────────────────────────────────────────

def run(target_angle=1.0, duration=5.0, render=True):
    # load model
    model = mujoco.MjModel.from_xml_path(
        os.path.join(os.path.dirname(__file__), "../plants/arm.xml")
    )
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # reset to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # controller
    pid = PID(kp=50.0, ki=10.0, kd=5.0, ctrl_min=-10, ctrl_max=10)

    # logging
    n_steps = int(duration / dt)
    log_time  = np.zeros(n_steps)
    log_q     = np.zeros(n_steps)
    log_qdot  = np.zeros(n_steps)
    log_ctrl  = np.zeros(n_steps)

    # sensor indices
    idx_q    = model.sensor("q").id
    idx_qdot = model.sensor("qdot").id

    # ── main loop ──────────────────────────────────────────────────

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for step in range(n_steps):
                # read sensors
                theta     = data.sensordata[idx_q]
                theta_dot = data.sensordata[idx_qdot]

                # compute control
                error = target_angle - theta
                tau   = pid.update(error, dt)

                # apply control
                data.ctrl[0] = tau

                # step physics
                mujoco.mj_step(model, data)
                viewer.sync()

                # log
                log_time[step] = data.time
                log_q[step]    = theta
                log_qdot[step] = theta_dot
                log_ctrl[step] = tau
    else:
        for step in range(n_steps):
            theta     = data.sensordata[idx_q]
            theta_dot = data.sensordata[idx_qdot]

            error = target_angle - theta
            tau   = pid.update(error, dt)
            data.ctrl[0] = tau

            mujoco.mj_step(model, data)

            log_time[step] = data.time
            log_q[step]    = theta
            log_qdot[step] = theta_dot
            log_ctrl[step] = tau

    # ── plot ───────────────────────────────────────────────────────

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(log_time, log_q, label="theta")
    axes[0].axhline(target_angle, color="r", linestyle="--", label="target")
    axes[0].set_ylabel("angle (rad)")
    axes[0].legend()

    axes[1].plot(log_time, log_qdot, label="theta_dot")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].legend()

    axes[2].plot(log_time, log_ctrl, label="ctrl (torque)")
    axes[2].set_ylabel("torque (N·m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    plt.suptitle(f"PID step response — target = {target_angle:.2f} rad")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../outputs/step_response.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    run(target_angle=1.0, duration=5.0, render=True)
```

### How to run

```bash
cd pid-arm
pip install mujoco matplotlib numpy
python mujoco_experiments/sim.py
```

The viewer window shows the arm in 3D. After it finishes, a plot saves to `outputs/step_response.png` with three panels: angle tracking, velocity, and control effort over time.

### What to experiment with

**P-only** — set `ki=0, kd=0`. Watch it oscillate around the target and never settle (underdamped) or settle below the target (steady-state error from gravity).

**PD** — add `kd=5`. Oscillation damps out but steady-state error remains because gravity is a constant disturbance the P-term alone can't cancel.

**PID** — add `ki=10`. Integral winds up to cancel gravity. Watch for overshoot — if it's too aggressive, the integral saturates `ctrlrange` and you see windup.

**Kill gravity** — change `<option gravity="0 0 0"/>` in the XML. Now P-only should nail the target with no steady-state error. This isolates your controller from the gravity disturbance and confirms it works in the simple case.

**Anti-windup** — set a very high `ki` (like 100) and watch the integral term blow up, then verify the clamping logic in the PID class prevents it.