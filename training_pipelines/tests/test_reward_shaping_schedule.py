from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from training_utils.reward_shaping import ShapedReward


class DummyEnv(gym.Env):
    """Minimal Gymnasium environment for testing reward shaping."""

    metadata = {}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "capacity_utilisation": spaces.Box(
                    low=0.0,
                    high=2.0,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.MultiDiscrete([11, 11, 11])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        obs = {
            "capacity_utilisation": np.asarray(
                [0.50],
                dtype=np.float32,
            )
        }

        return obs, {}

    def step(self, action):
        obs = {
            "capacity_utilisation": np.asarray(
                [0.50],
                dtype=np.float32,
            )
        }

        info = {
            "demand": np.asarray(
                [10, 10, 10],
                dtype=np.float32,
            ),
            "fulfilled_demand": np.asarray(
                [10, 10, 10],
                dtype=np.float32,
            ),
            "order_quantities": np.asarray(
                [0, 0, 0],
                dtype=np.int64,
            ),
        }

        return obs, 0.0, False, False, info


def test_shaping_reaches_zero_at_local_budget():
    env = ShapedReward(
        DummyEnv(),
        anneal_steps=5,
    )

    env.reset(seed=123)

    last_info = None

    for _ in range(5):
        _, _, _, _, last_info = env.step(
            np.asarray([0, 0, 0])
        )

    assert last_info is not None
    assert last_info["anneal_factor"] == 0.0
    assert last_info["shaping"] == 0.0


def test_vectorized_anneal_budget_math():
    total_training_steps = 200_000
    n_envs = 4

    local_anneal_steps = int(
        np.ceil(total_training_steps / max(1, n_envs))
    )

    assert local_anneal_steps == 50_000


def test_single_env_budget_is_unchanged():
    total_training_steps = 150_000
    n_envs = 1

    local_anneal_steps = int(
        np.ceil(total_training_steps / max(1, n_envs))
    )

    assert local_anneal_steps == 150_000