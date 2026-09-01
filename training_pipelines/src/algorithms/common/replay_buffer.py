
"""Replay buffer storing full SARSA transitions (Phase 4, section 7.3).

Storing the actual next action taken by the behavior policy (not re-derived
at sample time) is what preserves the SARSA target when using a replay
buffer instead of pure online updates.
"""
from __future__ import annotations

import numpy as np


class SarsaReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_actions = np.zeros(self.capacity, dtype=np.int64)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_action: int,
        done: bool,
    ) -> None:
        idx = self._cursor
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._next_actions[idx] = next_action
        self._dones[idx] = float(done)
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_obs[indices],
            "next_actions": self._next_actions[indices],
            "dones": self._dones[indices],
        }