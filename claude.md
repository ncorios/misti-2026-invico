# MISTI 2026 — Quadruped Controller Comparison

## Project
Controller comparison study on a 12-DOF quadruped in MuJoCo. Goal: implement and
benchmark PID, MPC, and PPO controllers on standard dog locomotion. A perception layer
(voice/video → LLM → command API) is a likely later phase, not this sprint — keep the
command interface clean enough to bolt it on later. NOT chameleon gait — normal quadruped
walking. A separate masters student owns the pure-RL chameleon work; stay out of that lane.

## Controller interface (locked — do not change)
controller(command: Command, obs: Obs) -> joint_targets: np.ndarray(12)
All three controllers conform to this signature so they're swappable behind one benchmark
harness. The future perception/LLM layer will produce Command objects — design Command
to be extensible but don't build that layer yet.

## Stack
- Sim: MuJoCo (Python).
- MPC: TWO candidate paths, decide in week 1 based on build cost vs. speed:
    (a) google-deepmind/mujoco_mpc (MJPC) — ships a quadruped task, but it's a C++/GUI
        app needing a from-source cmake/ninja build and Python bindings work. Powerful
        but the build + integration can eat a day+.
    (b) Compact iLQR or predictive-sampling MPC in Python directly against the MuJoCo
        env. Lighter, keeps all 3 controllers in one language behind one interface.
        Likely faster for a 1-week comparison. See arxiv 2503.04613 for the iLQR baseline.
  NEITHER is installed yet. Evaluate (b) first unless there's a reason to need MJPC's GUI.
- RL: PPO. Train in sim, log reward + tracking metrics. Use an existing implementation
    (stable-baselines3 or spinning-up) rather than writing PPO from scratch.
- Robot model: position-actuator path (matches hardware). Keep torque model separate.
- Sim-to-real is a stretch goal, not a current commitment.

## Status
- PID baseline: DONE. ~14 mrad RMSE avg across joints in sim.
- MPC: building this week.
- PPO: next week.

## Working style
Move fast. Prefer existing scaffolding over building from scratch. Get a working baseline
running first, optimize second. Flag where I'm reinventing something a library already
does. Keep all controllers behind the shared interface so I can benchmark head-to-head
without rewiring. When a step has a build/integration risk (e.g. MJPC), say so before
sinking time into it.
