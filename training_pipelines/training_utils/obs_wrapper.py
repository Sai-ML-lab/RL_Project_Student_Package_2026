
"""Observation-flattening wrapper and standalone helper.

The helper `flatten_observation` is deliberately dependency-free so that each
submitted `policy.py` can copy it verbatim without importing this package.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Product-level constants mirror the ones exposed by the environment.
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)

# Feature layout (kept as documentation and to detect drift):
# 0:3   inventory / 100
# 3:15  arrival_pipeline / 100 (row-major)
# 15:18 mean of last 3 days of demand / reference mean
# 18:21 mean of last 7 days of demand / reference mean
# 21:24 std of last 7 days of demand / reference mean
# 24:25 day / 50
# 25:26 capacity_utilisation
# 26:29 inventory position (inventory + pipeline sum) / 200
# 29:32 inventory position in volume / capacity
# 32:35 (inventory position - 3-day-mean-demand) / 100  (safety cushion signal)
FEATURE_DIM = 35


def flatten_observation(observation) -> np.ndarray:
    """Convert the environment observation dict to a float32 feature vector.

    Duplicated in each policy file so that submissions do not depend on this
    package at evaluation time.
    """
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    last3_mean = demand_history[-3:].mean(axis=0)
    last7_mean = demand_history.mean(axis=0)
    last7_std = demand_history.std(axis=0)

    inv_position_units = inventory + pipeline.sum(axis=1)
    inv_position_volume = inv_position_units * _PRODUCT_VOLUMES

    features = np.concatenate(
        [
            inventory / 100.0,
            pipeline.reshape(-1) / 100.0,
            last3_mean / _REF_DEMAND_MEANS,
            last7_mean / _REF_DEMAND_MEANS,
            last7_std / _REF_DEMAND_MEANS,
            np.asarray([day / 50.0], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
            inv_position_units / 200.0,
            inv_position_volume / _CAPACITY,
            (inv_position_units - last3_mean) / 100.0,
        ],
        dtype=np.float32,
    )
    return features.astype(np.float32, copy=False)


class FlattenObs(gym.ObservationWrapper):
    """Observation wrapper that produces a fixed-size float32 Box observation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return flatten_observation(observation)