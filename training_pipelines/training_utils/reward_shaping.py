
"""Reward shaping wrapper for training.

The wrapper strictly adds signals derived from the same `info` dict already
returned by the environment. The baseline reward is preserved.

Coefficients are annealed linearly to zero across the specified number of
environment steps so the final training regime matches the official reward.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ShapedReward(gym.Wrapper):
    """Add differentiable, monotonically-annealed shaping to the base reward.

    Shaping components (per timestep):
      + alpha * (service_level - 1.0)                # penalise stockouts extra
      - beta  * max(0, capacity_utilisation - 0.85)  # discourage discards
      - gamma * n_orders_placed / 3                  # gently favour batching
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.15,
        anneal_steps: int = 300_000,
    ) -> None:
        super().__init__(env)
        self.alpha_start = float(alpha)
        self.beta_start = float(beta)
        self.gamma_start = float(gamma)
        self.anneal_steps = max(1, int(anneal_steps))
        self._global_step = 0

    def _anneal_factor(self) -> float:
        frac = 1.0 - self._global_step / self.anneal_steps
        return float(max(0.0, frac))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_step += 1
        anneal = self._anneal_factor()

        # Service level per product; guard against zero demand days.
        demand = np.asarray(info["demand"], dtype=np.float32)
        fulfilled = np.asarray(info["fulfilled_demand"], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            service = np.where(demand > 0, fulfilled / demand, 1.0)
        service_level = float(service.mean())

        # Post-step capacity_utilisation is available on the returned observation.
        capacity_utilisation = float(
            np.asarray(obs["capacity_utilisation"]).reshape(-1)[0]
        )
        n_orders = float(np.count_nonzero(info["order_quantities"]))

        shaping = (
            self.alpha_start * (service_level - 1.0)
            - self.beta_start * max(0.0, capacity_utilisation - 0.85)
            - self.gamma_start * (n_orders / 3.0)
        ) * anneal

        info = dict(info)
        info["base_reward"] = reward
        info["shaping"] = shaping
        info["anneal_factor"] = anneal
        return obs, reward + shaping, terminated, truncated, info