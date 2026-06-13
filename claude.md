## Project
Controller comparison study on a 12-DOF quadruped in MuJoCo. Goal: implement and
benchmark PID, MPC, and PPO controllers on standard dog locomotion. A perception layer
(voice/video → LLM → command API) is a likely later phase, not this sprint — keep the
command interface clean enough to bolt it on later. NOT chameleon gait — normal quadruped
walking.

## Pre-publication checklist (run before final writeup / submission)
- Consider adding trained model weights (controllers/ppo/models/, *.zip) to the repo so
  readers can reproduce results without retraining — update .gitignore to un-ignore them.
- General repo cleanup: remove dead code, tidy directory structure.
- Write/update README with setup instructions, how to run each controller, benchmark results.
  Videos can be embedded inline in the README (e.g. via GitHub's video upload or a YouTube
  link) rather than committing .mp4 files to the repo.
- Bug sweep: run all three controllers end-to-end, check for silent failures.
- Regenerate requirements.txt from a clean environment (pip freeze after fresh install).
- Review .gitignore — make sure no result files (metrics.json, summary.csv) are excluded.

## Controller interface (locked — do not change)
controller(command: Command, obs: Obs) -> joint_targets: np.ndarray(12)
All three controllers conform to this signature so they're swappable behind one benchmark
harness. The interface is also the seam where a future perception/LLM command layer plugs
in (it would produce Command objects). Design Command to be extensible but don't build
that layer yet.