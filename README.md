# DIRECTORY #

misti-2026-invico/
├── configs/        # all the numbers: gait params, measured hardware limits, RL settings
├── models/         # MuJoCo robot files + meshes (your old plants/)
├── src/chameleon/  # the main code: gait, controller, safety, the shared contract
│   ├── rl/         # training only — torch/gym live here
│   └── deploy/     # runs on the Pi — pure numpy, no torch
├── perception/     # James: voice, vision, LLM router, xgolib
├── scripts/        # things you run by hand
├── tests/          # safety checks
├── data/           # chameleon video, labels, reference curves
├── checkpoints/    # trained policy weights
├── notes/          # theory + reading notes
└── sandbox/        # throwaway experiments + the old torque PID