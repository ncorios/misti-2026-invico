# MISTI 2026 — Quadruped Controller Comparison

Controller comparison study on a 12-DOF quadruped (XGO) in MuJoCo.
Goal: implement and benchmark PID, MPC, and PPO on standard dog locomotion.

## Status
| Controller | Status |
|------------|--------|
| PID        | Done — ~14 mrad RMSE avg across joints |
| MPC        | In progress |
| PPO        | Implemented — v1 model trained, eval results in `controllers/ppo/ppo_eval/` |

## Directory

```
misti-2026-invico/
├── controllers/
│   ├── pid/
│   │   ├── pid.py              # PID controller
│   │   ├── log_pid_run.py      # run + log a PID episode
│   │   ├── plot_pid_log.py     # plot logged run data
│   │   ├── Testpid.py          # test harness
│   │   └── controler-v1.xml    # MuJoCo model used for PID
│   ├── mpc/
│   │   └── goon.py             # MPC controller (in progress)
│   └── ppo/
│       ├── dogzilla_env.py     # custom Gymnasium env (12-DOF XGO)
│       ├── ppo_controller.py   # PPO controller (wraps trained policy)
│       ├── ppo_training.py     # training script (SB3 PPO)
│       ├── ppo_log.py          # logging utilities
│       ├── assets/             # XGO mesh STLs + ppo_dog.xml
│       ├── models/             # saved model weights (*.zip)
│       └── ppo_eval/           # eval outputs: metrics.json, episode mp4s, summary.csv
├── envs/
│   ├── Prueba2_19_03.xml       # shared MuJoCo scene
│   ├── assets/XGO/             # XGO mesh STLs
│   └── hardware.yaml           # physical robot spec
├── benchmark/
│   └── fall_detection.py       # fall detection utility
├── results/                    # PID run logs and plots
├── perception/
│   ├── vision.py               # vision stub (future LLM command layer)
│   └── voice-control.py        # voice stub
├── chameleon/                  # separate RL/deploy project — do not touch
└── notes/                      # theory and MuJoCo reference notes
```

## Controller interface

All three controllers share the same signature so they're swappable behind one benchmark harness:

```python
controller(command: Command, obs: Obs) -> joint_targets: np.ndarray  # shape (12,)
```

## Quick start

```bash
# PID
python controllers/pid/log_pid_run.py
python controllers/pid/plot_pid_log.py

# PPO — train
python controllers/ppo/ppo_training.py

# PPO — eval (loads models/dog_ppo_v1.zip by default)
python controllers/ppo/ppo_controller.py
```
