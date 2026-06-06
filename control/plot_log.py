"""
plot_log.py — plot desired vs actual joint angles and tracking error.

Reads run_log.npz (written by log_run.py) from the same folder and saves two
figures next to it:
    tracking.png   desired vs actual, 4 legs x 3 joints
    error.png      tracking error over time, grouped by joint type

    python plot_log.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # comment this out if you want interactive windows
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "run_log.npz"))
t, des, act, err = d["t"], d["desired"], d["actual"], d["error"]

legs = ["LF", "RF", "LH", "RH"]
cols = ["abduction", "hip", "knee"]
DESIRED_C, ACTUAL_C = "#2ca02c", "#1f77b4"

# ── Figure 1: desired vs actual, 4 legs x 3 joints ──────────────────────────────
fig, ax = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
for r, leg in enumerate(legs):
    for c, col in enumerate(cols):
        j = r * 3 + c
        a = ax[r, c]
        a.plot(t, des[:, j], "--", color=DESIRED_C, lw=1.6, label="desired")
        a.plot(t, act[:, j], "-",  color=ACTUAL_C,  lw=1.4, label="actual")
        rms = np.sqrt(np.mean(err[:, j] ** 2)) * 1e3
        a.set_title(f"{leg} {col}   (RMS {rms:.1f} mrad)", fontsize=10)
        a.grid(alpha=0.3)
        if c == 0:
            a.set_ylabel("angle (rad)")
        if r == 3:
            a.set_xlabel("time (s)")
ax[0, 0].legend(loc="upper right", fontsize=9)
fig.suptitle("desired vs actual joint angles", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(HERE, "tracking.png"), dpi=150)

# ── Figure 2: error over time, grouped by joint type ────────────────────────────
fig2, ax2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
leg_colors = {"LF": "#1f77b4", "RF": "#ff7f0e", "LH": "#2ca02c", "RH": "#9467bd"}
for c, col in enumerate(cols):
    a = ax2[c]
    a.axhline(0, color="k", lw=0.8)
    for r, leg in enumerate(legs):
        a.plot(t, err[:, r * 3 + c], color=leg_colors[leg], lw=1.3, label=leg)
    grp = err[:, [r * 3 + c for r in range(4)]]
    rms = np.sqrt(np.mean(grp ** 2)) * 1e3
    a.set_title(f"{col} joints — tracking error (group RMS {rms:.1f} mrad)", fontsize=11)
    a.set_ylabel("error (rad)")
    a.grid(alpha=0.3)
    a.legend(loc="upper right", ncol=4, fontsize=9)
ax2[-1].set_xlabel("time (s)")
fig2.suptitle("tracking error (desired − actual) over time", fontsize=13, y=0.995)
fig2.tight_layout(rect=[0, 0, 1, 0.98])
fig2.savefig(os.path.join(HERE, "error.png"), dpi=150)

print("saved tracking.png and error.png in", HERE)
