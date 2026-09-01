
"""Action-space wrapper built on the canonical codec (Phase 1, section 4.1 / 2).

Does not touch observations, rewards, or transition dynamics -- only recodes
the action the agent sees, per the "do not modify the official environment
logic" constraint.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_codec import (
    JOINT_ACTION_SIZE,
    joint_index_to_multidiscrete,
    multidiscrete_to_joint_index,
)


class JointActionWrapper(gym.ActionWrapper):
    """Expose ``Discrete(1331)`` to the agent; the env still receives ``MultiDiscrete([11,11,11])``.

    Required for DQN-family algorithms, whose standard implementations only
    support a single ``Discrete`` action space.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(JOINT_ACTION_SIZE)

    def action(self, action):
        indices = joint_index_to_multidiscrete(int(action))
        return np.asarray(indices, dtype=np.int64)

    def reverse_action(self, action):
        return multidiscrete_to_joint_index(np.asarray(action, dtype=np.int64).tolist())