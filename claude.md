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

### PID TRACK — FINDING (for writeup; honest framing)
- The PID controller is solid: tracks the commanded joint trajectory to ~14 mrad RMS,
  stays upright, stable across gait frequencies. The controller does its job.
- BUT the reference it's handed is NOT a locomotion gait. It's an open-loop kinematic
  oscillator (CPG-style sinusoids, diagonal trot pairing) with no notion of ground contact
  or reaction force. Tracking joint angles ≠ propulsion. Result: the dog barely translates
  (~0.008 m/s, ~0.1 m over 12 s at the best frequency).
- Measured failure mode (freq=1.0, headless diag): feet only clear ~2–4 cm (no real swing
  phase), so they stay on the ground and SLIDE — the in-stance x-slide ≈ the full foot
  x-stroke (e.g. LF 64 of 65 mm). Forward and backward scuffs nearly cancel: body oscillates
  ±7 cm but nets only ~5 cm in 6 s. A properly planted foot should stay fixed in the world
  while the hip retracts to drive the body forward; here it scuffs in place.
- Frequency is only cadence, not stroke geometry: cranking it scuffs faster, doesn't travel
  farther. 2 Hz was actually WORSE than 1 Hz in the sweep (feet can't complete the stroke).
- WHY it matters for the comparison: this is the clean illustration of the thesis. A good
  controller + a hand-authored reference still doesn't walk, because it has no model of
  contact/foot-placement. That's exactly what MPC supplies explicitly (contact schedule +
  foot trajectories) and PPO learns implicitly from reward. PID's honest contribution is the
  tracking baseline, NOT locomotion — don't claim the PID "walks."
- Gap to close if we ever want PID to actually walk: a contact-aware gait generator
  (guaranteed swing clearance + no-slip stance where planted-foot world velocity ≈ 0).
  PI recommended this oscillator model; it's a reference-trajectory limitation, not a PID one.

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

#### CURRENT STATE (live — keep updated each working session)
_Last updated: 2026-06-24._
- **Best model: v24** (15.30m det / 0.90 survival; 15.56m stoch / 0.90). Warmstarted from v0
  with the new energy penalty on — a clean win on every axis (beats v0 14.30/0.70 and the
  lost v21). Energy term cleaned the gait AND improved distance + survival. v0 is the
  prior best / fallback; v21 was best-ever (~15.5–16m, 100%) but its .zip was deleted. v23
  COLLAPSED to 0.83m (stood still) from over-escalated heading cost — heading is powerful but
  freezes walking if overweighted.
- **Two open problems, distinct fixes:** (1) drift/straightness — over ~15m, y-spread ±1–2m,
  deterministic episodes fan out BOTH directions from reset noise → policy isn't closing the
  loop on heading (it could; orientation quat is in obs). (2) gait quality — v24 now GALLOPS
  (pitch oscillation: tilt modest ~11° mean, but pitch RATE large ~2.7 rad/s). Energy term
  fixed dragging but pushed toward a ballistic bound.
- **Decisions (2026-06-22):**
  - Gait efficiency via ENERGY penalty (chosen over memory-based phase-symmetry, which bakes
    in a pace-like 2-beat gait not a natural trot — steer to phase-clock-in-obs if phasing
    ever needed). DONE → v24.
  - Anti-gallop = ANGLE-based leveling term, NOT a pitch-RATE cost. A rate cost repeats the
    turning_cost mistake (punishes corrective pitching, can't allow recovery). Use a lin+quad
    PITCH-ANGLE cost (reuse the heading lin+quad trick to bite near zero). Documented as a
    FUTURE implementation in ppo_eval/v24/v24-notes.md; not yet coded.
- **In progress:** user tuning y_drift + heading weights for straightness. NOTE: heading is
  the stronger lever than y_drift (policy can OBSERVE heading via quat but NOT y — y is
  privileged/excluded, so y_drift can't drive active closed-loop correction, only weak
  open-loop selection pressure). Heading = cause, y_drift = symptom. y_drift's unique value:
  catches "crabbing" (facing +x but sliding sideways) that heading misses.
- **Caveat to keep honest:** cm-level drift over 15m is not reachable by reward shaping
  (~1° heading error ≈ 0.26m drift over 15m); absolute-y can never be closed-loop-corrected
  since y-position is (correctly) excluded from obs as privileged — only heading is. Realistic
  target ~0.2–0.5m. Energy magnitude is non-physical (placeholder forcerange ~50× too high) —
  a RELATIVE penalty, fine for in-sim comparison.
- **Eval instrumentation (DONE 2026-06-24):** ppo_log.py now logs per-run y-drift (final_y,
  mean|y|), heading bias + mean|yaw| (deg), AND overlays a single deterministic, NO-RESET-NOISE
  run on the trajectory plot to isolate the policy's own heading from reset spread. summary.csv
  upserts by version. (Cost-of-transport still TODO.)

### TO EXPLORE BEFORE AMP (parked — low priority right now)
- **Suspected bad XML masses/inertias.** Strong belief the joint/link masses + inertias in the
  XML are wrong (XGO debug-placeholder comments are the evidence; it's not our authored XML —
  to be edited later). This is the leading hypothesis for any systematic gait weirdness
  (gallop, heading lean) and must be ruled out before blaming reward shaping or moving to AMP.
  Already in HONEST LIMITATIONS, escalated here as an action item.
- **Heading bias vs noise — UNRESOLVED.** The no-reset-noise eval (2026-06-24) showed most
  capable policies' single deterministic run curving the SAME sign (~−8° to −17° yaw), which
  *looked* like a structural bias (→ would point back at the XML). BUT user's read: the drift
  is essentially NOISE-DRIVEN and random, not structural — v41 in particular shows basically no
  drift. n=1 per model on the no-noise run is thin evidence; multi-episode deterministic still
  fans out BOTH directions from reset noise (consistent with the noise-driven view). Don't treat
  the consistent-sign no-noise bias as established. Resolve (more no-noise samples / seeds) only
  if drift actually matters for the writeup; otherwise leave it.
- Next phase intent: AMP (adversarial motion priors) for gait quality, superseding hand-shaped
  reward terms.

### HONEST LIMITATIONS (do NOT overclaim)
- XML dynamics (masses, inertias, kp, forcerange) are approximate XGO debug placeholders,
  NOT measured DOGZILLA values — fine for in-sim comparison, need system-ID for hardware.
  STRONGER: joint/link masses + inertias are likely outright wrong (debug-comment evidence,
  inherited XML) — suspect this before reward shaping for any gait pathology. See TO EXPLORE.
- Sim-to-real not yet attempted; if done, report the gap honestly.
- The comparison is the contribution; "RL solved locomotion from scratch" is NOT the claim.