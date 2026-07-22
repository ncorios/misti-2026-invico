"""
ppo_controller.py — wrap a trained PPO policy as a benchmark controller.

Adapts a saved SB3 PPO model to the shared controller interface used across the
comparison study:

    controller(command, obs) -> joint_targets: np.ndarray(12)

so a learned policy is swappable with the PID/MPC controllers behind one benchmark
harness. The returned action is DogEnv's residual joint target (added to the stand
pose inside the env), already shape (12,). `command` is accepted for interface
compatibility and is currently unused (the policy is goal-agnostic).

This is a library wrapper imported by the harness, not a run-directly script; use
ppo_watch.py to view a policy or ppo_log.py to evaluate one.
"""
from stable_baselines3 import PPO


class PPOController:
    """Loads a trained PPO model and calls it as controller(command, obs) -> action(12)."""

    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def __call__(self, command, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action   # already np.ndarray(12), already the joint targets
