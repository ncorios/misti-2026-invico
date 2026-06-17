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

## Project Writeup Notes (MISTI / INVICO)

### PROJECT FRAMING
- INVICO Lab internship (Univ. Panamericana, MISTI). Controller comparison study on a
  DOGZILLA S2 quadruped: PID → MPC → PPO, evaluated in MuJoCo; sim-to-real + demo are
  stretch goals.
- Scope history: originally a chameleon-inspired DeepMimic task, rescoped by the PI to
  the controller comparison. PID baseline built fast (<14 mrad RMSE tracking); that
  credibility was used to renegotiate the scope.
- OWNERSHIP (keep strict, never blur in writeup):
  - Mine: PPO controller track (RL env, training, eval, results).
  - James (Course 16): MPC track + autonomy stack (Whisper→LLM command router,
    vision/face-following).
  - Shared seam: controller(command, obs)->joint_targets(12) interface contract; my
    contribution at that boundary is the contract, not James's autonomy code.

### COMPARISON STUDY (the deliverable)
- Three controllers, one interface, compared on TASK-LEVEL metrics (forward velocity,
  distance, cost of transport, fall rate, robustness under perturbation) — not internal
  metrics, because PID/MPC track references (native metric = tracking RMSE) and PPO
  optimizes reward; they don't share an internal score. Deliberate methodological choice.
- Fall rate only discriminates under perturbation, not flat ground.

### PPO TRACK (mine)
- Custom MuJoCo Gym env from the Ant-v5 skeleton, redefined for the real robot.
- Design decisions, each defensible:
  - Obs restricted to hardware-measurable (IMU + joint angles); excluded base linear
    velocity, joint velocities, contact forces (sim-only → sim-to-real gap). Privileged
    info allowed in reward, never in obs.
  - Residual action space (small offsets around stand pose) — THE unlock for learning to
    walk. Diagnosed via a "hold stand pose, no policy" test isolating plant from
    controller, proving the pose was stable so the failure was exploration destabilizing it.
  - Position control throughout (matches real servos), not torque.
  - Orientation-based fall termination (rotate up-axis, check world-z) after finding a
    reward exploit: dog lay on its back farming the survival bonus.
- Results (use distance/survival, NOT reward_mean — not comparable across versions):
  v9 best — 7.91m deterministic / 70% survival over 1000 steps; 5.77m / 40% stochastic.
  Speed/stability tradeoff documented (v8 faster-but-falls vs v9 balanced). Stochastic
  retention higher on better models (v9 ~73% vs v7 ~13%) — better policies more robust.
- Methodology rigor: single-episode eval was noisy and gave a WRONG conclusion (thought
  v9 regressed; multi-episode revealed it's best). Switched to multi-episode dual-mode
  eval (deterministic = deployment, stochastic = robustness). Reward shaping done as a
  diagnostic loop (failure mode → which term → one-variable change → retrain → compare);
  documented failures: lunging at high forward weight, standing-still at high cost
  weights, circling. Planned ablation: does stability/turning help or hurt vs v9's
  simpler reward — early data suggests simpler may win; report honestly.

### HONEST LIMITATIONS (do NOT overclaim)
- XML dynamics (masses, inertias, kp, forcerange) are approximate XGO debug placeholders,
  NOT measured DOGZILLA values — fine for in-sim comparison, need system-ID for hardware.
- Sim-to-real not yet attempted; if done, report the gap honestly.
- The comparison is the contribution; "RL solved locomotion from scratch" is NOT the claim.