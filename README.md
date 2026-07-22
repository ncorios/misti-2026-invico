# MISTI 2026 — Quadruped Controller Comparison

Controller comparison study on a 12-DOF quadruped (XGO) in MuJoCo.
Goal: implement and benchmark PID, MPC, and PPO on standard dog locomotion.

## Status
| Controller | Status |
|------------|--------|
| PID        | Done — ~16 mrad RMSE avg across joints |
| MPC        | In progress |
| PPO        | Implemented — v1 model trained, eval results in `controllers/ppo/ppo_eval/` |

## Directory

```
dogzilla-control/
├── controllers/
│   ├── pid/
│   │   ├── pid.py              # PID controller class (P/I/D + anti-windup + feedforward)
│   │   ├── Testpid.py          # run the PID gait in the MuJoCo viewer
│   │   ├── log_pid_run.py      # run headless, log joint/desired/error arrays -> run_log.npz
│   │   ├── plot_pid_log.py     # plot tracking + error from the logged run
│   │   └── controler-v1.xml    # MuJoCo model for PID (torque / <motor> actuators)
│   ├── mpc/
│   │   └── mpc.py              # MPC controller — separate repo, not yet merged (stub)
│   └── ppo/
│       ├── dogzilla_env.py     # custom Gymnasium env (12-DOF DOGZILLA S2 / XGO)
│       ├── ppo_training.py     # train PPO from scratch (SB3)
│       ├── ppo_warmstart.py    # continue / fine-tune training from a saved version
│       ├── ppo_watch.py        # watch a trained policy in the MuJoCo viewer
│       ├── ppo_log.py          # dual-mode eval -> metrics.json, trajectory.png, video
│       ├── ppo_controller.py   # wrap a trained policy as controller(command, obs) -> (12)
│       ├── stand_test.py       # no-policy plant sanity check (holds the stand keyframe)
│       ├── ppo-progress-log.md # per-version training + reward-shaping log
│       ├── assets/             # ppo_dog.xml (position actuators) + XGO mesh STLs
│       ├── models/             # saved model weights (*.zip, gitignored)
│       ├── ppo_eval/           # eval outputs: metrics.json, trajectory.png, mp4, summary.csv
│       └── tb_logs/            # TensorBoard training logs (gitignored)
├── envs/
│   ├── Prueba2_19_03.xml       # shared MuJoCo scene
│   ├── assets/XGO/             # XGO mesh STLs
│   ├── hardware.yaml           # physical robot spec (stub)
│   └── chameleon/              # separate CAD deliverable — do not touch
├── notes/                      # controller theory + MuJoCo reference notes
├── benchmark/
│   └── benchmark.py            # task-level benchmark harness (pending MPC — stub)
├── results/                    # PID run log (run_log.npz) + tracking/error plots
├── requirements.txt
└── README.md
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
