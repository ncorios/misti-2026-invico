# MISTI 2026 — Quadruped Controller Comparison

Controller comparison study on a 12-DOF quadruped (XGO Dogzilla S2) in MuJoCo.
Goal: implement and benchmark PID, MPC, and PPO on standard dog locomotion.

## Status
| Controller | Status |
|------------|--------|
| PID        | Done: ~16 mrad RMSE avg across joints |
| MPC        | In progress |
| PPO        | Implemented: v41 model trained with a straight gallop, eval results in `controllers/ppo/ppo_eval/` |

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
│   │   └── mpc.py              # MPC controller: separate repo, not yet merged (stub)
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
│   ├── Prueba2_19_03.xml       # Original MuJoCo File - from advisor
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

## Quick start

```bash
# PID — run a tracking episode, then plot it
python controllers/pid/log_pid_run.py      # -> run_log.npz
python controllers/pid/plot_pid_log.py     # -> tracking.png, error.png

# PPO — reproduce the best policy's eval (weights committed: v41)
python controllers/ppo/ppo_log.py 41           # metrics + trajectory plot (+ video)
mjpython controllers/ppo/ppo_watch.py 41       # live viewer (macOS: mjpython, else python)

# PPO — train from scratch / fine-tune (optional)
python controllers/ppo/ppo_training.py <version> [--steps N] [--envs K]
python controllers/ppo/ppo_warmstart.py <from_version> <to_version> [--steps N] [--envs K]
```
