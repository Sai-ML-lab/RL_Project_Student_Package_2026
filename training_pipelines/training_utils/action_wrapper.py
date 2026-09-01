
"""Action-flattening wrapper mapping Discrete(1331) <-> MultiDiscrete([11,11,11]).

Enables SB3 DQN (which supports only Discrete action spaces) to operate on the
project's multi-discrete action space. The base-11 encoding is inline-copyable
by each submitted `policy.py`.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_SPACE_SIZE = 11 * 11 * 11  # 1331


def joint_index_to_quantities(index: int) -> list[int]:
    """Decode a joint action index into three order quantities in {0,10,...,100}.

    Kept dependency-free so it can be copied verbatim into each policy file.
    """
    index = int(index)
    a2 = index % 11
    index //= 11
    a1 = index % 11
    a0 = index // 11
    return [a0 * 10, a1 * 10, a2 * 10]


def _quantities_to_joint_index(action_indices: np.ndarray) -> int:
    a0, a1, a2 = (int(x) for x in action_indices)
    return a0 * 121 + a1 * 11 + a2


class FlattenAction(gym.ActionWrapper):
    """Wrap MultiDiscrete([11,11,11]) as Discrete(1331) using base-11 encoding."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def action(self, action):
        quantities = joint_index_to_quantities(int(action))
        return np.asarray(
            [q // 10 for q in quantities], dtype=np.int64
        )

    def reverse_action(self, action):
        return _quantities_to_joint_index(np.asarray(action, dtype=np.int64))