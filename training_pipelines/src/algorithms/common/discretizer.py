
"""Sparse coarse-coded feature discretizer for TD(lambda)/Tabular SARSA (Phase 5, section 8.1).

Turns the observation into "bands" (low/medium/high, encoded 0/1/2) instead
of raw continuous values, then one-hot-encodes each band-group and
concatenates them into one sparse binary feature vector -- a simple, fast
form of coarse coding with a small, fixed number of active features per
state (no full state table required).

Per-product bands (5 x 3 products = 15): inventory position coverage,
on-hand coverage, pipeline coverage, demand level, demand trend.
Global bands (3): capacity utilisation, projected capacity utilisation,
episode phase.
"""
from __future__ import annotations

import numpy as np

from industrial_inventory_env.environment import IndustrialInventoryEnv

_REFERENCE_DEMAND_MEANS = IndustrialInventoryEnv.REFERENCE_DEMAND_MEANS.astype(np.float32)
_REFERENCE_LEAD_TIMES = IndustrialInventoryEnv.REFERENCE_LEAD_TIMES.astype(np.float32)
_PRODUCT_VOLUMES = IndustrialInventoryEnv.PRODUCT_VOLUMES.astype(np.float32)
_CAPACITY = float(IndustrialInventoryEnv.CAPACITY)
_HORIZON = float(IndustrialInventoryEnv.HORIZON)
_NUM_PRODUCTS = IndustrialInventoryEnv.NUM_PRODUCTS
_EPS = 1e-6

N_BINS = 3  # low / medium / high, per band-group

PER_PRODUCT_BAND_NAMES = (
    "inventory_position_coverage",
    "on_hand_coverage",
    "pipeline_coverage",
    "demand_level",
    "demand_trend",
)
GLOBAL_BAND_NAMES = ("capacity_utilisation", "projected_capacity_utilisation", "episode_phase")
N_BAND_GROUPS = len(PER_PRODUCT_BAND_NAMES) * _NUM_PRODUCTS + len(GLOBAL_BAND_NAMES)  # 18
FEATURE_DIM = N_BAND_GROUPS * N_BINS  # 54


def _coverage_band(days_covered: float, lead_time: float) -> int:
    if days_covered < lead_time:
        return 0
    if days_covered < 2.0 * lead_time:
        return 1
    return 2


def _demand_level_band(recent_mean: float, reference_mean: float) -> int:
    if recent_mean < 0.9 * reference_mean:
        return 0
    if recent_mean > 1.1 * reference_mean:
        return 2
    return 1


def _trend_band(trend: float) -> int:
    if trend < -2.0:
        return 0
    if trend > 2.0:
        return 2
    return 1


def _utilisation_band(value: float) -> int:
    if value < 0.5:
        return 0
    if value < 0.85:
        return 1
    return 2


def discretize_observation(observation) -> tuple[int, ...]:
    """Return one band index (0/1/2) per band-group, in a fixed order."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])

    pipeline_total = pipeline.sum(axis=1)
    inventory_position = inventory + pipeline_total
    last3_mean = demand_history[-3:].mean(axis=0)
    last7_mean = demand_history.mean(axis=0)
    trend = last3_mean - last7_mean
    demand_estimate = np.maximum(last3_mean, _EPS)

    bands: list[int] = []
    for product in range(_NUM_PRODUCTS):
        lead_time = float(_REFERENCE_LEAD_TIMES[product])
        bands.append(_coverage_band(float(inventory_position[product] / demand_estimate[product]), lead_time))
        bands.append(_coverage_band(float(inventory[product] / demand_estimate[product]), lead_time))
        bands.append(_coverage_band(float(pipeline_total[product] / demand_estimate[product]), lead_time))
        bands.append(_demand_level_band(float(last3_mean[product]), float(_REFERENCE_DEMAND_MEANS[product])))
        bands.append(_trend_band(float(trend[product])))

    current_volume = float(np.dot(inventory, _PRODUCT_VOLUMES))
    pipeline_volume = float(np.dot(pipeline_total, _PRODUCT_VOLUMES))
    projected_utilisation = (current_volume + pipeline_volume) / _CAPACITY
    phase = day / _HORIZON
    episode_phase = 0 if phase < (1.0 / 3.0) else (1 if phase < (2.0 / 3.0) else 2)

    bands.append(_utilisation_band(capacity_utilisation))
    bands.append(_utilisation_band(projected_utilisation))
    bands.append(episode_phase)

    return tuple(bands)


def encode_features(bands: tuple[int, ...]) -> np.ndarray:
    """One-hot encode each band and concatenate into a sparse (FEATURE_DIM,) vector."""
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    for group_index, band in enumerate(bands):
        features[group_index * N_BINS + band] = 1.0
    return features


def flatten_observation_bands(observation) -> np.ndarray:
    """Discretize + one-hot encode in one call."""
    return encode_features(discretize_observation(observation))