"""Representation C: compact demand/lead-time/capacity-aware features.

Representation C extends the existing 76-D Representation B without using any
information outside the official observation.  It deliberately stays small
(~100 dimensions) and focuses on signals that matter directly for inventory
control:

* EWMA demand
* demand min/max and slope
* coefficient of variation
* lead-time demand uncertainty
* safety stock
* weekly phase (sin/cos)

Early-episode zero padding is excluded from all demand statistics.
"""
from __future__ import annotations

from math import pi

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engineered import (
    REPRESENTATION_B_DIM,
    flatten_observation_representation_b,
)
from .observation import _DEFAULT if False else RAW_FEATURE_DIM

# Physical constants mirror the official environment and Representation B.
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_HORIZON = 50.0
_REFERENCE_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.float32)
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_NUM_PRODUCTS = 3
_HISTORY_DAYS = 7
_EPS = 1e-6

# Seven additions per product x 3 products = 21; two global weekly features.
C_PER_PRODUCT_FEATURE_NAMES = (
    "ewma_demand",
    "demand_slope",
    "demand_cv",
    "recent_min_demand",
    "recent_max_demand",
    "lead_time_demand_uncertainty",
    "safety_stock",
)
C_PER_PRODUCT_FEATURE_DIM = len(C_PER_PRODUCT_FEATURE_NAMES) * _NUM_PRODUCTS
C_GLOBAL_FEATURE_NAMES = ("weekly_phase_sin", "weekly_phase_cos")
C_GLOBAL_FEATURE_DIM = len(C_GLOBAL_FEATURE_NAMES)
REPRESENTATION_C_ADDITION_DIM = C_PER_PRODUCT_FEATURE_DIM + C_GLOBAL_FEATURE_DIM
REPRESENTATION_C_DIM = REPRESENTATION_B_DIM + REPRESENTATION_C_ADDITION_DIM  # 99


def _valid_history(demand_history: np.ndarray, day: int) -> np.ndarray:
    valid_days = min(max(day, 0), _HISTORY_DAYS)
    if valid_days <= 0:
        return np.zeros((0, _NUM_PRODUCTS), dtype=np.float32)
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
    denom = float(np.sum(x_centered * x_centered))
    y_centered = history - history.mean(axis=0, keepdims=True)
    return (np.sum(x_centered[:, None] * y_centered, axis=0) / max(denom, _EPS)).astype(
        np.float32
    )


def compute_representation_c_additions(observation) -> np.ndarray:
    """Return the 23 features that are appended to Representation B."""
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    history = _valid_history(demand_history, day)

    if len(history) == 0:
        ewma = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        slope = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        cv = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        recent_min = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        recent_max = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        std = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    else:
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        ewma = _ewma(history)
        slope = _slope(history)
        cv = np.clip(std / np.maximum(mean, 1.0), 0.0, 3.0)
        recent_min = history.min(axis=0)
        recent_max = history.max(axis=0)

    lead_time_demand_uncertainty = std * np.sqrt(_REFERENCE_LEAD_TIMES)
    # Conservative base-stock signal: 95%-style safety stock using observed
    # demand variability over the reference lead time.  It is a state feature,
    # not an externally imposed action rule.
    safety_stock = 1.65 * lead_time_demand_uncertainty

    per_product = np.stack(
        [
            ewma / _REF_DEMAND_MEANS,
            slope / _REF_DEMAND_MEANS,
            cv,
            recent_min / _REF_DEMAND_MEANS,
            recent_max / _REF_DEMAND_MEANS,
            lead_time_demand_uncertainty / 100.0,
            safety_stock / 100.0,
        ],
        axis=1,
    ).reshape(-1)

    phase = 2.0 * pi * (day % 7) / 7.0
    global_features = np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)

    return np.concatenate([per_product.astype(np.float32, copy=False), global_features])


def flatten_observation_representation_c(observation) -> np.ndarray:
    """Return Representation B + Representation C additions (99-D)."""
    features = np.concatenate(
        [flatten_observation_representation_b(observation), compute_representation_c_additions(observation)]
    ).astype(np.float32, copy=False)
    if features.shape != (REPRESENTATION_C_DIM,):
        raise RuntimeError(
            f"Representation C shape mismatch: {features.shape}; expected {(REPRESENTATION_C_DIM,)}"
        )
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Representation C produced non-finite features")
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
