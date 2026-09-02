"""A2C Representation C submission for the IITM RL Inventory Control project.

Candidate: V030 / A2C / Representation C
The policy is self-contained so the portal ZIP only needs policy.py and model.zip.
"""
from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C

_NUM_PRODUCTS = 3
_HISTORY_DAYS = 7
_REFERENCE_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.float32)
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_EPS = 1e-6


def _flatten_observation_representation_b(observation) -> np.ndarray:
    """Return the exact 76-D Representation B used by the training pipeline."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    valid_days = min(max(day, 0), _HISTORY_DAYS)
    history = (
        demand_history[-valid_days:].astype(np.float32, copy=False)
        if valid_days > 0
        else np.empty((0, _NUM_PRODUCTS), dtype=np.float32)
    )

    if len(history) == 0:
        recent_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        recent_std = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        trend = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        last3_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    else:
        recent_mean = history.mean(axis=0)
        recent_std = history.std(axis=0)
        last3 = history[-3:]
        last3_mean = last3.mean(axis=0)
        if len(history) <= 1:
            trend = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        else:
            x = np.arange(len(history), dtype=np.float32)
            xc = x - x.mean()
            denom = float(np.sum(xc * xc))
            yc = history - history.mean(axis=0, keepdims=True)
            trend = np.sum(xc[:, None] * yc, axis=0) / max(denom, _EPS)

    pipeline_total = pipeline.sum(axis=1)
    inventory_position = inventory + pipeline_total
    lead_time_demand = recent_mean * _REFERENCE_LEAD_TIMES
    pipeline_within_lead = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    for p in range(_NUM_PRODUCTS):
        lead = int(_REFERENCE_LEAD_TIMES[p])
        pipeline_within_lead[p] = pipeline[p, : min(lead, pipeline.shape[1])].sum()
    pipeline_after_lead = pipeline_total - pipeline_within_lead
    inv_position_gap = inventory_position - lead_time_demand
    days_of_supply = inventory_position / np.maximum(recent_mean, 1.0)

    global_features = np.asarray(
        [capacity_utilisation, day / 50.0, (day % 7) / 7.0], dtype=np.float32
    )

    features = np.concatenate(
        [
            inventory / 100.0,
            pipeline.reshape(-1) / 100.0,
            last3_mean / _REF_DEMAND_MEANS,
            recent_mean / _REF_DEMAND_MEANS,
            recent_std / _REF_DEMAND_MEANS,
            trend / _REF_DEMAND_MEANS,
            pipeline_within_lead / 100.0,
            pipeline_after_lead / 100.0,
            lead_time_demand / 100.0,
            inventory_position / 200.0,
            inv_position_gap / 100.0,
            days_of_supply / 10.0,
            inventory_position * _PRODUCT_VOLUMES / _CAPACITY,
            global_features,
        ],
        dtype=np.float32,
    )

    if features.shape != (76,):
        raise ValueError(f"Representation B shape mismatch: {features.shape}; expected (76,)")
    return features


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
    xc = x - x.mean()
    denominator = float(np.sum(xc * xc))
    yc = history - history.mean(axis=0, keepdims=True)
    return (np.sum(xc[:, None] * yc, axis=0) / max(denominator, _EPS)).astype(np.float32)


def _representation_c(observation) -> np.ndarray:
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    valid_days = min(max(day, 0), _HISTORY_DAYS)
    history = (
        demand_history[-valid_days:].astype(np.float32, copy=False)
        if valid_days > 0
        else np.empty((0, _NUM_PRODUCTS), dtype=np.float32)
    )

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
