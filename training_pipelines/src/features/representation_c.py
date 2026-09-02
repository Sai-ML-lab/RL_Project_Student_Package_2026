"""Representation C: compact demand/lead-time-aware features.

Representation C extends Representation B (76-D) with a small set of
observation-only features that are useful for inventory control.  No future
demand, info fields, hidden state, or leaderboard-specific data are used.

The output is 99-D:
    76-D Representation B + 23 additional features.

Early-episode zero padding in demand_history is excluded from demand
statistics, matching Representation B's intended semantics.
"""
from __future__ import annotations

from math import pi

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engineered import REPRESENTATION_B_DIM, flatten_observation_representation_b

_NUM_PRODUCTS = 3
_HISTORY_DAYS = 7
_REFERENCE_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.float32)
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_EPS = 1e-6

C_PER_PRODUCT_FEATURE_NAMES = (
    "ewma_demand",
    "demand_slope",
    "demand_cv",
    "recent_min_demand",
    "recent_max_demand",
    "lead_time_demand_uncertainty",
    "safety_stock",
)
C_PER_PRODUCT_FEATURE_DIM = len(C_PER_PRODUCT_FEATURE_NAMES) * _NUM_PRODUCTS  # 21
C_GLOBAL_FEATURE_NAMES = ("weekly_phase_sin", "weekly_phase_cos")
C_GLOBAL_FEATURE_DIM = len(C_GLOBAL_FEATURE_NAMES)  # 2
REPRESENTATION_C_ADDITION_DIM = C_PER_PRODUCT_FEATURE_DIM + C_GLOBAL_FEATURE_DIM  # 23
REPRESENTATION_C_DIM = REPRESENTATION_B_DIM + REPRESENTATION_C_ADDITION_DIM  # 99


def _valid_history(demand_history: np.ndarray, day: int) -> np.ndarray:
    valid_days = min(max(day, 0), _HISTORY_DAYS)
    if valid_days == 0:
        return np.empty((0, _NUM_PRODUCTS), dtype=np.float32)
    return demand_history[-valid_days:].astype(np.float32, copy=False)


def _ewma(history: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    value = history[0].copy()
    for row in history[1:]:
        value = alpha * row + (1.0 - alpha) * value
    return value.astype(np.float32, copy=False)


def _slope(history: np.ndarray) -> np.ndarray:
    if len(history) <= 1:
        return np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    x = np.arange(len(history), dtype=np.float32)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered * x_centered))
    y_centered = history - history.mean(axis=0, keepdims=True)
    return (
        np.sum(x_centered[:, None] * y_centered, axis=0) / max(denominator, _EPS)
    ).astype(np.float32)


def compute_representation_c_additions(observation) -> np.ndarray:
    """Compute the 23 new Representation C features."""
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    history = _valid_history(demand_history, day)

    if len(history) == 0:
        ewma = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        slope = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        demand_cv = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        recent_min = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        recent_max = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        std = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    else:
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        ewma = _ewma(history)
        slope = _slope(history)
        demand_cv = np.clip(std / np.maximum(mean, 1.0), 0.0, 3.0)
        recent_min = history.min(axis=0)
        recent_max = history.max(axis=0)

    lead_time_demand_uncertainty = std * np.sqrt(_REFERENCE_LEAD_TIMES)
    safety_stock = 1.65 * lead_time_demand_uncertainty

    per_product = np.stack(
        [
            ewma / _REF_DEMAND_MEANS,
            slope / _REF_DEMAND_MEANS,
            demand_cv,
            recent_min / _REF_DEMAND_MEANS,
            recent_max / _REF_DEMAND_MEANS,
            lead_time_demand_uncertainty / 100.0,
            safety_stock / 100.0,
        ],
        axis=1,
    ).reshape(-1)

    phase = 2.0 * pi * (day % 7) / 7.0
    weekly_phase = np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)

    additions = np.concatenate([per_product.astype(np.float32, copy=False), weekly_phase])
    if additions.shape != (REPRESENTATION_C_ADDITION_DIM,):
        raise RuntimeError(
            f"Representation C additions shape mismatch: {additions.shape}; "
            f"expected {(REPRESENTATION_C_ADDITION_DIM,)}"
        )
    if not np.all(np.isfinite(additions)):
        raise FloatingPointError("Representation C additions contain non-finite values")
    return additions


def flatten_observation_representation_c(observation) -> np.ndarray:
    """Return the complete 99-D Representation C vector."""
    features = np.concatenate(
        [flatten_observation_representation_b(observation), compute_representation_c_additions(observation)]
    ).astype(np.float32, copy=False)
    if features.shape != (REPRESENTATION_C_DIM,):
        raise RuntimeError(
            f"Representation C shape mismatch: {features.shape}; expected {(REPRESENTATION_C_DIM,)}"
        )
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Representation C produced non-finite values")
    return features


class RepresentationCObsWrapper(gym.ObservationWrapper):
    """Gymnasium wrapper producing the fixed-size Representation C vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(REPRESENTATION_C_DIM,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return flatten_observation_representation_c(observation)
