# MISTI 2026 — Quadruped Controller Comparison

Controller comparison study on a 12-DOF quadruped (XGO) in MuJoCo.
Goal: implement and benchmark PID, MPC, and PPO on standard dog locomotion.

## Status
| Controller | Status |
|------------|--------|
| PID        | Done — ~14 mrad RMSE avg across joints |
| MPC        | In progress |
| PPO        | Up next |

## Directory

```
misti-2026-invico/
├── controllers/    # PID, MPC, PPO — all share controller(command, obs) -> np.ndarray(12)
├── envs/           # MuJoCo XML models, mesh assets, hardware.yaml
├── benchmark/      # run_pid.py, log_run.py, plot_log.py, fall_detection.py
├── results/        # logged run data and plots
├── perception/     # voice + vision stubs (future LLM command layer)
├── chameleon/      # separate RL/deploy work (different project, do not touch)
└── notes/          # theory and MuJoCo reference notes
```

## Controller interface

All three controllers conform to the same signature so they're swappable behind one benchmark harness:

```python
controller(command: Command, obs: Obs) -> joint_targets: np.ndarray  # shape (12,)
```

## Running the PID benchmark

```bash
python benchmark/run_pid.py
python benchmark/plot_log.py
```
