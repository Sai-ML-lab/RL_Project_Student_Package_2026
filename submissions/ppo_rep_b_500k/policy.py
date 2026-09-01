"""Frozen PPO + Representation B submission candidate.

Technique:
    Proximal Policy Optimization (PPO)

Representation:
    Representation B = 38 raw normalized features + 38 engineered features
    = 76 features.

This file is deliberately self-contained for leaderboard inference.  The
preprocessing below mirrors training_pipelines/src/features/observation.py
and training_pipelines/src/features/engineered.py exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Constants mirrored from the official environment / Representation B.
# ---------------------------------------------------------------------------

_PRODUCT_VOLUMES = np.asarray(
    [2.0, 3.0, 1.5],
    dtype=np.float32,
)

_CAPACITY = 1000.0
_HORIZON = 50.0
_REFERENCE_LEAD_TIMES = np.asarray(
    [3, 2, 1],
    dtype=np.int64,
)

_NUM_PRODUCTS = 3
_DEMAND_HISTORY_DAYS = 7

# Exact DEFAULT_SCALES from features.normalizer.py.
_INVENTORY_SCALE = 200.0
_PIPELINE_SCALE = 100.0
_DEMAND_HISTORY_SCALE = 100.0
_DAY_SCALE = 49.0


def _flatten_observation_raw(observation) -> np.ndarray:
    """Exact 38-D Representation-A preprocessing used by Representation B."""
    inventory = np.asarray(
        observation["inventory"],
        dtype=np.float32,
    )
    pipeline = np.asarray(
        observation["arrival_pipeline"],
        dtype=np.float32,
    )
    demand_history = np.asarray(
        observation["demand_history"],
        dtype=np.float32,
    )
    day = float(
        np.asarray(observation["day"]).reshape(-1)[0]
    )
    capacity_utilisation = float(
        np.asarray(
            observation["capacity_utilisation"]
        ).reshape(-1)[0]
    )

    features = np.concatenate(
        [
            inventory / _INVENTORY_SCALE,
            pipeline.reshape(-1) / _PIPELINE_SCALE,
            demand_history.reshape(-1) / _DEMAND_HISTORY_SCALE,
            np.asarray(
                [day / _DAY_SCALE],
                dtype=np.float32,
            ),
            np.asarray(
                [capacity_utilisation],
                dtype=np.float32,
            ),
        ]
    )

    return features.astype(np.float32, copy=False)


def _flatten_observation_engineered(observation) -> np.ndarray:
    """Exact 38-D engineered feature construction used during training."""
    inventory = np.asarray(
        observation["inventory"],
        dtype=np.float32,
    )
    pipeline = np.asarray(
        observation["arrival_pipeline"],
        dtype=np.float32,
    )
    demand_history = np.asarray(
        observation["demand_history"],
        dtype=np.float32,
    )
    day = int(
        np.asarray(observation["day"]).reshape(-1)[0]
    )

    inventory_position = inventory + pipeline.sum(axis=1)

    # Ignore zero padding before enough real demand-history days exist.
    valid_days = min(
        day,
        _DEMAND_HISTORY_DAYS,
    )

    if valid_days == 0:
        last3_mean = np.zeros(
            _NUM_PRODUCTS,
            dtype=np.float32,
        )
        last7_mean = np.zeros(
            _NUM_PRODUCTS,
            dtype=np.float32,
        )
        demand_std = np.zeros(
            _NUM_PRODUCTS,
            dtype=np.float32,
        )
    else:
        valid_history = demand_history[-valid_days:]
        valid3 = valid_history[-min(valid_days, 3):]
        last3_mean = valid3.mean(axis=0)
        last7_mean = valid_history.mean(axis=0)
        demand_std = valid_history.std(axis=0)

    demand_trend = last3_mean - last7_mean

    within_lead_time = np.zeros(
        _NUM_PRODUCTS,
        dtype=np.float32,
    )
    after_lead_time = np.zeros(
        _NUM_PRODUCTS,
        dtype=np.float32,
    )

    for product in range(_NUM_PRODUCTS):
        lead_time = int(
            _REFERENCE_LEAD_TIMES[product]
        )
        within_lead_time[product] = (
            pipeline[product, :lead_time].sum()
        )
        after_lead_time[product] = (
            pipeline[product, lead_time:].sum()
        )

    lead_time_demand_estimate = (
        last7_mean * _REFERENCE_LEAD_TIMES
    )
    inventory_position_gap = (
        inventory_position - lead_time_demand_estimate
    )
    estimated_days_of_supply = (
        inventory_position
        / np.maximum(last7_mean, 1.0)
    )

    per_product = np.stack(
        [
            inventory_position,
            last3_mean,
            last7_mean,
            demand_trend,
            demand_std,
            within_lead_time,
            after_lead_time,
            lead_time_demand_estimate,
            inventory_position_gap,
            estimated_days_of_supply,
        ],
        axis=1,
    )

    global_features = _compute_global_features(
        inventory,
        pipeline,
        day,
    )

    return np.concatenate(
        [
            per_product.reshape(-1).astype(
                np.float32,
                copy=False,
            ),
            global_features,
        ]
    ).astype(np.float32, copy=False)


def _compute_global_features(
    inventory: np.ndarray,
    pipeline: np.ndarray,
    day: int,
) -> np.ndarray:
    """Exact 8 global engineered features used during training."""
    current_volume = float(
        np.dot(
            inventory,
            _PRODUCT_VOLUMES,
        )
    )
    pipeline_volume = float(
        np.dot(
            pipeline.sum(axis=1),
            _PRODUCT_VOLUMES,
        )
    )

    current_volume_ratio = (
        current_volume / _CAPACITY
    )
    headroom_ratio = (
        max(_CAPACITY - current_volume, 0.0)
        / _CAPACITY
    )
    pipeline_volume_ratio = (
        pipeline_volume / _CAPACITY
    )
    projected_utilisation = (
        current_volume + pipeline_volume
    ) / _CAPACITY
    remaining_fraction = (
        max(_HORIZON - day, 0.0)
        / _HORIZON
    )

    phase = day / _HORIZON
    early = 1.0 if phase < (1.0 / 3.0) else 0.0
    middle = (
        1.0
        if (1.0 / 3.0) <= phase < (2.0 / 3.0)
        else 0.0
    )
    late = 1.0 if phase >= (2.0 / 3.0) else 0.0

    return np.asarray(
        [
            current_volume_ratio,
            headroom_ratio,
            pipeline_volume_ratio,
            projected_utilisation,
            remaining_fraction,
            early,
            middle,
            late,
        ],
        dtype=np.float32,
    )


def _flatten_observation_representation_b(observation) -> np.ndarray:
    """Return exact 76-D Representation B."""
    features = np.concatenate(
        [
            _flatten_observation_raw(observation),
            _flatten_observation_engineered(observation),
        ]
    ).astype(np.float32, copy=False)

    if features.shape != (76,):
        raise ValueError(
            f"Expected Representation B shape (76,), got {features.shape}"
        )

    return features


_MODEL_PATH = Path(__file__).resolve().parent / "model.zip"
_MODEL = PPO.load(
    str(_MODEL_PATH),
    device="cpu",
)


def run_policy(observation):
    """Deterministic PPO inference.

    SB3 returns MultiDiscrete([11, 11, 11]) action indices.  The official
    environment interprets index i as an order quantity of 10*i.
    """
    features = _flatten_observation_representation_b(observation)

    action, _ = _MODEL.predict(
        features,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.int64,
    ).reshape(-1)

    if action.shape != (3,):
        raise ValueError(
            f"Expected action shape (3,), got {action.shape}"
        )

    if np.any(action < 0) or np.any(action > 10):
        raise ValueError(
            f"Invalid PPO action indices: {action.tolist()}"
        )

    quantities = (
        action * 10
    ).tolist()

    return [int(q) for q in quantities]
