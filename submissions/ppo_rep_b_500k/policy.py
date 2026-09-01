"""Frozen PPO Representation B submission candidate.

Technique:
    Proximal Policy Optimization (PPO)

Representation:
    Representation B = 38 raw normalized features
                       + 38 engineered features
                       = 76 features

The preprocessing is kept self-contained so the submission does not depend
on the training package at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Representation-B constants mirrored from the official environment.
# ---------------------------------------------------------------------------

_PRODUCT_VOLUMES = np.asarray(
    [2.0, 3.0, 1.5],
    dtype=np.float32,
)

_CAPACITY = 1000.0
_HORIZON = 50.0

_REFERENCE_LEAD_TIMES = np.asarray(
    [3, 4, 2],
    dtype=np.float32,
)

_NUM_PRODUCTS = 3
_DEMAND_HISTORY_DAYS = 7

_EPS = 1e-6

# These are the normalization scales used by Representation A.
_INVENTORY_SCALE = 100.0
_PIPELINE_SCALE = 100.0
_DEMAND_HISTORY_SCALE = 100.0
_DAY_SCALE = 50.0


def _flatten_observation_raw(observation) -> np.ndarray:
    """Return the exact 38-D normalized raw representation."""

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

    return features.astype(
        np.float32,
        copy=False,
    )


def _compute_per_product_features(observation) -> np.ndarray:
    """Return the 30 engineered per-product features."""

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

        valid3 = valid_history[
            -min(valid_days, 3):
        ]

        last3_mean = valid3.mean(axis=0)
        last7_mean = valid_history.mean(axis=0)
        demand_std = valid_history.std(axis=0)

    demand_trend = (
        last3_mean - last7_mean
    )

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
        inventory_position
        - lead_time_demand_estimate
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

    return per_product.reshape(-1).astype(
        np.float32
    )


def _compute_global_features(observation) -> np.ndarray:
    """Return the 8 engineered global features."""

    inventory = np.asarray(
        observation["inventory"],
        dtype=np.float32,
    )

    pipeline = np.asarray(
        observation["arrival_pipeline"],
        dtype=np.float32,
    )

    day = float(
        np.asarray(observation["day"]).reshape(-1)[0]
    )

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

    late = (
        1.0
        if phase >= (2.0 / 3.0)
        else 0.0
    )

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
    """Return the exact 76-D Representation B vector."""

    raw = _flatten_observation_raw(observation)

    engineered = np.concatenate(
        [
            _compute_per_product_features(observation),
            _compute_global_features(observation),
        ]
    ).astype(np.float32)

    features = np.concatenate(
        [raw, engineered]
    ).astype(
        np.float32,
        copy=False,
    )

    if features.shape != (76,):
        raise ValueError(
            f"Expected Representation B shape (76,), "
            f"got {features.shape}"
        )

    return features


# ---------------------------------------------------------------------------
# Load model once at import time.
# ---------------------------------------------------------------------------

_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model.zip"
)

_MODEL = PPO.load(
    str(_MODEL_PATH),
    device="cpu",
)


def run_policy(observation):
    """Run deterministic PPO inference.

    The SB3 MultiDiscrete([11, 11, 11]) action is converted from indices
    0..10 into the required quantities 0..100 in increments of 10.
    """

    features = _flatten_observation_representation_b(
        observation
    )

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
            f"Model produced invalid action indices: {action.tolist()}"
        )

    quantities = (
        action * 10
    ).astype(
        np.int64
    ).tolist()

    if any(
        q < 0 or q > 100 or q % 10 != 0
        for q in quantities
    ):
        raise ValueError(
            f"Invalid order quantities: {quantities}"
        )

    return [int(q) for q in quantities]
