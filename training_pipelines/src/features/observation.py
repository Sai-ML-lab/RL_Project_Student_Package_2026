
"""Representation A: raw, normalized observation features (Phase 1, section 4.2).

Fixed flattening order (38 raw values total):
    inventory (3), arrival_pipeline (12), demand_history (21), day (1),
    capacity_utilisation (1)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .normalizer import DEFAULT_SCALES, NormalizationScales

RAW_FEATURE_DIM = 38


def flatten_observation_raw(
    observation, scales: NormalizationScales = DEFAULT_SCALES
) -> np.ndarray:
    """Flatten and normalize the official observation dict into a (38,) float32 vector."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    features = np.concatenate(
        [
            inventory / scales.inventory_scale,
            pipeline.reshape(-1) / scales.pipeline_scale,
            demand_history.reshape(-1) / scales.demand_history_scale,
            np.asarray([day / scales.day_scale], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
        ]
    )
    return features.astype(np.float32, copy=False)


class RawObsWrapper(gym.ObservationWrapper):
    """Observation wrapper producing the fixed-size Representation A vector."""

    def __init__(self, env: gym.Env, scales: NormalizationScales = DEFAULT_SCALES) -> None:
        super().__init__(env)
        self._scales = scales
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(RAW_FEATURE_DIM,), dtype=np.float32
        )

    def observation(self, observation):
        return flatten_observation_raw(observation, self._scales)