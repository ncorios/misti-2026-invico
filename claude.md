## Project
Controller comparison study on a 12-DOF quadruped in MuJoCo. Goal: implement and
benchmark PID, MPC, and PPO controllers on standard dog locomotion. A perception layer
(voice/video → LLM → command API) is a likely later phase, not this sprint — keep the
command interface clean enough to bolt it on later. NOT chameleon gait — normal quadruped
walking.

## Controller interface (locked — do not change)
controller(command: Command, obs: Obs) -> joint_targets: np.ndarray(12)
All three controllers conform to this signature so they're swappable behind one benchmark
harness. The interface is also the seam where a future perception/LLM command layer plugs
in (it would produce Command objects). Design Command to be extensible but don't build
that layer yet.