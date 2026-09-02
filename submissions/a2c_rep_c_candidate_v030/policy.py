"""A2C Representation C submission for the IITM RL Inventory Control project.

Candidate: V030 / A2C / Representation C.
The policy is self-contained; the portal ZIP only needs policy.py and model.zip.
"""
from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C

_NUM_PRODUCTS = 3
_HISTORY_DAYS = 7
_HORIZON = 50.0
_REFERENCE_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.float32)
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_EPS = 1e-6


def _flatten_observation_raw(observation) -> np.ndarray:
    """Exact 38-D raw representation from the training pipeline."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )
    return np.concatenate(
        [
            inventory / 200.0,
            pipeline.reshape(-1) / 100.0,
            demand_history.reshape(-1) / 100.0,
            np.asarray([day / 49.0], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def _flatten_observation_representation_b(observation) -> np.ndarray:
    """Exact 76-D Representation B: raw 38 + engineered 38."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    inventory_position = inventory + pipeline.sum(axis=1)

    valid_days = min(day, _HISTORY_DAYS)
    if valid_days == 0:
        last3_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        last7_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        demand_std = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    else:
        valid_history = demand_history[-valid_days:]
        valid3 = valid_history[-min(valid_days, 3):]
        last3_mean = valid3.mean(axis=0)
        last7_mean = valid_history.mean(axis=0)
        demand_std = valid_history.std(axis=0)

    demand_trend = last3_mean - last7_mean

    within_lead_time = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    after_lead_time = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    for product in range(_NUM_PRODUCTS):
        lead_time = int(_REFERENCE_LEAD_TIMES[product])
        within_lead_time[product] = pipeline[product, :lead_time].sum()
        after_lead_time[product] = pipeline[product, lead_time:].sum()

    lead_time_demand_estimate = last7_mean * _REFERENCE_LEAD_TIMES
    inventory_position_gap = inventory_position - lead_time_demand_estimate
    estimated_days_of_supply = inventory_position / np.maximum(last7_mean, 1.0)

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
    ).reshape(-1).astype(np.float32)

    current_volume = float(np.dot(inventory, _PRODUCT_VOLUMES))
    pipeline_volume = float(np.dot(pipeline.sum(axis=1), _PRODUCT_VOLUMES))
    current_volume_ratio = current_volume / _CAPACITY
    headroom_ratio = max(_CAPACITY - current_volume, 0.0) / _CAPACITY
    pipeline_volume_ratio = pipeline_volume / _CAPACITY
    projected_utilisation = (current_volume + pipeline_volume) / _CAPACITY
    remaining_fraction = max(_HORIZON - float(day), 0.0) / _HORIZON
    phase = float(day) / _HORIZON
    early = 1.0 if phase < (1.0 / 3.0) else 0.0
    middle = 1.0 if (1.0 / 3.0) <= phase < (2.0 / 3.0) else 0.0
    late = 1.0 if phase >= (2.0 / 3.0) else 0.0
    global_features = np.asarray(
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

    features = np.concatenate([_flatten_observation_raw(observation), per_product, global_features])
    if features.shape != (76,):
        raise ValueError(f"Representation B shape mismatch: {features.shape}; expected (76,)")
    return features.astype(np.float32, copy=False)


def _valid_history(observation) -> tuple[np.ndarray, int]:
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    valid_days = min(max(day, 0), _HISTORY_DAYS)
    history = (
        demand_history[-valid_days:].astype(np.float32, copy=False)
        if valid_days > 0
        else np.empty((0, _NUM_PRODUCTS), dtype=np.float32)
    )
    return history, day


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


def _representation_c(observation) -> np.ndarray:
    """Exact 99-D Representation C from the training pipeline."""
    history, day = _valid_history(observation)
    base = _flatten_observation_representation_b(observation)

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

    features = np.concatenate([base, per_product.astype(np.float32, copy=False), weekly_phase])
    if features.shape != (99,):
        raise ValueError(f"Representation C shape mismatch: {features.shape}; expected (99,)")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Representation C produced non-finite features")
    return features.astype(np.float32, copy=False)


_MODEL_PATH = Path(__file__).resolve().parent / "model.zip"
_MODEL = A2C.load(str(_MODEL_PATH), device="cpu")


def run_policy(observation):
    features = _representation_c(observation)
    action, _ = _MODEL.predict(features, deterministic=True)
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    if action.size != 3 or np.any(action < 0) or np.any(action > 10):
        raise ValueError(f"Unexpected model action: {action.tolist()}")
    return [int(x * 10) for x in action]
